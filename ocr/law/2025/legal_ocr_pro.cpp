// legal_ocr_pro.cpp.cpp
// End to end pipeline for legal intake: PDF rasterization -> OCR -> deskew -> heuristics ->
// doc type classification -> compact prompt -> OpenAI function schema -> merge -> outputs.
// Features:
// - Medical records, Pleadings, Police reports, Transcripts, Insurance EOB, Imaging report
// - Snippet windows around keywords to minimize tokens
// - Parallel processing, rate limiting, retries with backoff
// - Cache by hash of snippet to avoid repeat API calls
// - Optional PII redaction in final JSON
// - Optional raw OCR auditing
// - Combined JSON, per file JSON, and JSONL export
//
// Build:
// g++ -std=c++17 -O2 -pthread \
//   -ltesseract -llept \
//   -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
//   -lcurl \
//   -o legal_ocr_pro legal_ocr_pro.cpp
//
// Usage:
// ./legal_ocr_pro INPUT_PATH OPENAI_API_KEY OUTPUT_JSON [--threads=N] [--lang=eng] [--model=gpt-4o-mini]
//    [--per-file] [--jsonl=path.jsonl] [--cache=.cache] [--redact] [--audit] [--timeout=120]
//    [--max-lines=14] [--max-chars=1400]

#include <filesystem>
#include <regex>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include <random>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>

#include <curl/curl.h>
#include "nlohmann_json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ---------------- Config defaults ----------------
struct Config {
    std::string input_path;
    std::string api_key;
    std::string output_json;
    std::string ocr_lang = "eng";
    std::string model = "gpt-4o-mini";
    std::string cache_dir;     // empty disables cache
    std::string jsonl_path;    // empty disables jsonl
    bool per_file = false;
    bool redact = false;
    bool audit_raw_ocr = false;
    int threads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    int http_timeout = 120; // seconds
    size_t max_snippet_lines = 14;
    size_t max_chars_per_snippet = 1400;
};

// ---------------- Helpers ----------------
static void die(const std::string &m) { std::cerr << "Error: " << m << std::endl; std::exit(1); }

static bool has_ext(const fs::path &p, std::initializer_list<std::string> exts) {
    if (!p.has_extension()) return false;
    auto e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    for (auto &x : exts) if (e == x) return true;
    return false;
}
static bool is_pdf(const fs::path &p)   { return has_ext(p, {".pdf"}); }
static bool is_image(const fs::path &p) { return has_ext(p, {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}); }

static std::string to_lower(std::string s) { std::transform(s.begin(), s.end(), s.begin(), ::tolower); return s; }
static std::string trim_copy(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// FNV-1a 64 bit for cache keys
static uint64_t fnv1a_64(const std::string &s) {
    const uint64_t FNV_OFFSET = 1469598103934665603ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;
    for (unsigned char c : s) { h ^= c; h *= FNV_PRIME; }
    return h;
}

// ---------------- CLI ----------------
static Config parse_cli(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " INPUT_PATH OPENAI_API_KEY OUTPUT_JSON [--threads=N] [--lang=eng] "
                  << "[--model=gpt-4o-mini] [--per-file] [--jsonl=path.jsonl] [--cache=.cache] "
                  << "[--redact] [--audit] [--timeout=120] [--max-lines=14] [--max-chars=1400]\n";
        std::exit(1);
    }
    Config c;
    c.input_path = argv[1];
    c.api_key = argv[2];
    c.output_json = argv[3];
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--threads=",0)==0) c.threads = std::max(1, std::stoi(a.substr(10)));
        else if (a.rfind("--lang=",0)==0) c.ocr_lang = a.substr(7);
        else if (a.rfind("--model=",0)==0) c.model = a.substr(8);
        else if (a == "--per-file") c.per_file = true;
        else if (a.rfind("--jsonl=",0)==0) c.jsonl_path = a.substr(8);
        else if (a.rfind("--cache=",0)==0) c.cache_dir = a.substr(8);
        else if (a == "--redact") c.redact = true;
        else if (a == "--audit") c.audit_raw_ocr = true;
        else if (a.rfind("--timeout=",0)==0) c.http_timeout = std::max(30, std::stoi(a.substr(10)));
        else if (a.rfind("--max-lines=",0)==0) c.max_snippet_lines = std::max<size_t>(6, std::stoul(a.substr(12)));
        else if (a.rfind("--max-chars=",0)==0) c.max_chars_per_snippet = std::max<size_t>(500, std::stoul(a.substr(12)));
    }
    return c;
}

// ---------------- Shell and curl ----------------
static int run_cmd(const std::string &cmd) { return std::system(cmd.c_str()); }

static size_t curl_write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static json http_post_json(const std::string &url, const std::string &bearer, const json &payload, long &http_code, int timeout_sec) {
    CURL *curl = curl_easy_init();
    if (!curl) die("curl init failed");
    std::string response;
    struct curl_slist *headers = nullptr;
    std::string auth = "Authorization: Bearer " + bearer;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string body = payload.dump();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        die(std::string("curl failed: ") + curl_easy_strerror(res));
    }

    try {
        return json::parse(response.empty() ? "{}" : response);
    } catch (...) {
        std::cerr << "Raw response: " << response << std::endl;
        die("Failed to parse JSON response from API");
    }
    return {};
}

// ---------------- PDF to images ----------------
static std::vector<std::string> pdf_to_images(const std::string &pdf_path, const std::string &out_dir_base) {
    fs::create_directories(out_dir_base);
    std::string prefix = (fs::path(out_dir_base) / "page").string();
    std::string cmd = "pdftoppm -png \"" + pdf_path + "\" \"" + prefix + "\"";
    int rc = run_cmd(cmd);
    if (rc != 0) die("pdftoppm failed for " + pdf_path);

    std::vector<std::string> paths;
    for (auto &entry : fs::directory_iterator(out_dir_base)) {
        if (entry.is_regular_file() && is_image(entry.path())) paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// ---------------- OCR with deskew ----------------
static cv::Mat deskew(const cv::Mat &srcGray) {
    // Basic deskew using Hough lines to estimate dominant angle
    cv::Mat bw;
    cv::adaptiveThreshold(srcGray, bw, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 31, 15);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(bw, lines, 1, CV_PI/180, 180);
    double angle_deg = 0.0;
    int count = 0;
    for (auto &l : lines) {
        float rho = l[0], theta = l[1];
        double deg = theta * 180.0 / CV_PI;
        // prefer near horizontal text baselines
        if (deg > 80 && deg < 100) continue;
        if (deg > 0 && deg < 45) { angle_deg += deg - 0; count++; }
        else if (deg > 135 && deg < 180) { angle_deg += deg - 180; count++; }
    }
    if (count == 0) return srcGray;
    angle_deg /= std::max(1, count);
    cv::Point2f center(srcGray.cols / 2.0f, srcGray.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);
    cv::Mat dst;
    cv::warpAffine(srcGray, dst, rot, srcGray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return dst;
}

static std::string ocr_image_path(const std::string &image_path, const Config &cfg) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) return "";
    cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray2 = deskew(gray);
    cv::Mat den; cv::fastNlMeansDenoising(gray2, den, 30.0);
    cv::Mat th;  cv::adaptiveThreshold(den, th, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 15);

    std::string tmp = (fs::temp_directory_path() / (fs::path(image_path).filename().string() + ".ocr.png")).string();
    cv::imwrite(tmp, th);

    tesseract::TessBaseAPI tess;
    if (tess.Init(nullptr, cfg.ocr_lang.c_str(), tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Tesseract init failed" << std::endl;
        return "";
    }
    tess.SetVariable("preserve_interword_spaces", "1");
    Pix *px = pixRead(tmp.c_str());
    if (!px) return "";
    tess.SetImage(px);
    char *out = tess.GetUTF8Text();
    std::string text = out ? std::string(out) : std::string();
    delete [] out;
    pixDestroy(&px);
    tess.End();
    return text;
}

// ---------------- Doc type classification ----------------
enum class DocType { MEDICAL, PLEADING, POLICE, TRANSCRIPT, INSURANCE_EOB, IMAGING, UNKNOWN };

static DocType classify_doc(const std::string &text) {
    std::string t = to_lower(text);
    int med=0, pld=0, pol=0, tr=0, eob=0, img=0;

    for (auto &k : { "diagnosis","treatment","medication","mrn","cpt","icd","history of present illness" }) if (t.find(k)!=std::string::npos) med++;
    for (auto &k : { "plaintiff","defendant","index no","caption","verified complaint","affirmation","affidavit","notice of motion","bill of particulars" }) if (t.find(k)!=std::string::npos) pld++;
    for (auto &k : { "police report","officer","badge","mv104","collision","accident report","precinct" }) if (t.find(k)!=std::string::npos) pol++;
    for (auto &k : { "examination before trial","ebt","deposition","q:","a:","court reporter","witness" }) if (t.find(k)!=std::string::npos) tr++;
    for (auto &k : { "explanation of benefits","eob","claim number","payer","allowed amount","denied","adjustment code"}) if (t.find(k)!=std::string::npos) eob++;
    for (auto &k : { "impression","findings","radiology","mri","ct","x-ray","ultrasound","images reviewed" }) if (t.find(k)!=std::string::npos) img++;

    int best = std::max({med,pld,pol,tr,eob,img});
    if (best==0) return DocType::UNKNOWN;
    if (best==med) return DocType::MEDICAL;
    if (best==pld) return DocType::PLEADING;
    if (best==pol) return DocType::POLICE;
    if (best==tr)  return DocType::TRANSCRIPT;
    if (best==eob) return DocType::INSURANCE_EOB;
    if (best==img) return DocType::IMAGING;
    return DocType::UNKNOWN;
}

static const char* doc_type_str(DocType d) {
    switch(d) {
        case DocType::MEDICAL: return "medical_record";
        case DocType::PLEADING: return "pleading";
        case DocType::POLICE: return "police_report";
        case DocType::TRANSCRIPT: return "transcript";
        case DocType::INSURANCE_EOB: return "insurance_eob";
        case DocType::IMAGING: return "imaging_report";
        default: return "unknown";
    }
}

// ---------------- Snippet extraction ----------------
static void add_keyword_windows(std::vector<std::string> &keep, const std::string &text,
                                const std::vector<std::string> &keys, size_t max_lines) {
    std::vector<std::string> lines;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) lines.push_back(trim_copy(line));

    auto tolow = [](std::string s){ std::transform(s.begin(), s.end(), s.begin(), ::tolower); return s; };
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string low = tolow(lines[i]);
        bool hit = false;
        for (auto &k : keys) if (low.find(k)!=std::string::npos) { hit = true; break; }
        if (hit) {
            size_t start = (i>=2? i-2 : 0);
            size_t end = std::min(lines.size(), i+3);
            for (size_t j = start; j < end; ++j) {
                if (!lines[j].empty()) keep.push_back(lines[j]);
                if (keep.size() >= max_lines) return;
            }
        }
    }
}

static std::string join_lines_trunc(const std::vector<std::string> &v, size_t max_chars) {
    std::string s;
    for (auto &l : v) {
        if (s.size() + l.size() + 1 > max_chars) break;
        s += l; s += "\n";
    }
    if (s.size() > max_chars) s.resize(max_chars);
    return s;
}

static json regex_first(const std::string &text, const std::regex &re) {
    std::smatch m;
    if (std::regex_search(text, m, re)) return json(m[0]);
    return nullptr;
}

static json local_extract_generic(const std::string &text) {
    json j;
    auto name = regex_first(text, std::regex(R"((?:Patient|Name)\s*[:\-]\s*([A-Za-z ,.\-']{3,90}))", std::regex::icase));
    if (!name.is_null()) j["name_candidate"] = name;
    auto date_any = regex_first(text, std::regex(R"((\b\d{4}-\d{2}-\d{2}\b)|(\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b))"));
    if (!date_any.is_null()) j["date_candidate"] = date_any;
    auto phone = regex_first(text, std::regex(R"((\+?\d{1,2}[\s\-\.])?(?:\(?\d{3}\)?[\s\-\.])?\d{3}[\s\-\.]\d{4})"));
    if (!phone.is_null()) j["phone_candidate"] = phone;
    return j;
}

static json local_extract_by_type(const std::string &text, DocType dt, const Config &cfg) {
    json j = local_extract_generic(text);

    std::vector<std::string> keys;
    if (dt == DocType::MEDICAL) {
        keys = {"diagnosis","dx","treatment","medication","procedure","impression","assessment","plan","chief complaint","history"};
    } else if (dt == DocType::PLEADING) {
        keys = {"caption","plaintiff","defendant","index no","cause of action","negligence","damages","wherefore","relief"};
    } else if (dt == DocType::POLICE) {
        keys = {"police report","officer","badge","mv104","collision","accident","location","vehicle","license","injury"};
    } else if (dt == DocType::TRANSCRIPT) {
        keys = {"q:","a:","examination before trial","deposition","witness","objection","page","line"};
    } else if (dt == DocType::INSURANCE_EOB) {
        keys = {"explanation of benefits","eob","payer","claim","allowed","denied","adjustment","remark code","member"};
    } else if (dt == DocType::IMAGING) {
        keys = {"impression","findings","technique","comparison","mri","ct","x-ray","ultrasound"};
    } else {
        keys = {"plaintiff","defendant","diagnosis","mv104","deposition","impression","eob"};
    }

    std::vector<std::string> keep;
    add_keyword_windows(keep, text, keys, cfg.max_snippet_lines);
    if (keep.empty()) {
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line) && keep.size() < cfg.max_snippet_lines) {
            line = trim_copy(line);
            if (!line.empty()) keep.push_back(line);
        }
    }
    j["important_snippets"] = join_lines_trunc(keep, cfg.max_chars_per_snippet);
    j["char_count"] = (int)text.size();

    // transcript page, line, quote extraction
    if (dt == DocType::TRANSCRIPT) {
        std::vector<json> cites;
        std::regex re_pg(R"(page\s+(\d+))", std::regex::icase);
        std::regex re_ln(R"(line[s]?\s+(\d+)(?:\s*-\s*(\d+))?)", std::regex::icase);
        std::istringstream iss(text);
        std::string line;
        int curPage = -1;
        while (std::getline(iss, line)) {
            std::smatch m;
            if (std::regex_search(line, m, re_pg)) curPage = std::stoi(m[1]);
            if (std::regex_search(line, m, re_ln)) {
                json c; c["page"] = std::max(0, curPage);
                c["line"] = m[0].str();
                c["text"] = trim_copy(line);
                cites.push_back(c);
                if (cites.size() >= 10) break;
            }
        }
        if (!cites.empty()) j["local_citations"] = cites;
    }

    return j;
}

// ---------------- Schemas ----------------
static json schema_medical() {
    return {
        {"name","extract_medical_json"},
        {"description","Return compact JSON for medical record"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"patient_name", {{"type","string"}}},
                {"dob", {{"type","string"}}},
                {"dates_of_service", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"diagnoses", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"procedures", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"medications", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"patient_name","confidence"})}
        }}
    };
}
static json schema_pleading() {
    return {
        {"name","extract_pleading_json"},
        {"description","Return compact JSON for pleading"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"court", {{"type","string"}}},
                {"caption", {{"type","string"}}},
                {"index_number", {{"type","string"}}},
                {"parties", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"causes_of_action", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"relief_sought", {{"type","string"}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"caption","confidence"})}
        }}
    };
}
static json schema_police() {
    return {
        {"name","extract_police_json"},
        {"description","Return compact JSON for police report"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"report_number", {{"type","string"}}},
                {"incident_date", {{"type","string"}}},
                {"location", {{"type","string"}}},
                {"officer", {{"type","string"}}},
                {"vehicles", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"injuries", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"violations", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"incident_date","confidence"})}
        }}
    };
}
static json schema_transcript() {
    return {
        {"name","extract_transcript_json"},
        {"description","Return compact JSON for deposition or 50-h transcript"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"witness_name", {{"type","string"}}},
                {"date", {{"type","string"}}},
                {"key_admissions", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"key_inconsistencies", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"credibility_factors", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"citations", {{"type","array"}, {"items", {
                    {"type","object"},
                    {"properties", {
                        {"page",{ {"type","integer"} }},
                        {"line",{ {"type","string"} }},
                        {"text",{ {"type","string"} }}
                    }},
                    {"required", json::array({"page","text"})}
                }}}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"confidence"})}
        }}
    };
}
static json schema_eob() {
    return {
        {"name","extract_eob_json"},
        {"description","Return compact JSON for insurance explanation of benefits"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"payer", {{"type","string"}}},
                {"member", {{"type","string"}}},
                {"claim_number", {{"type","string"}}},
                {"service_dates", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"allowed_amount", {{"type","string"}}},
                {"denied_amount", {{"type","string"}}},
                {"adjustments", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"payer","claim_number","confidence"})}
        }}
    };
}
static json schema_imaging() {
    return {
        {"name","extract_imaging_json"},
        {"description","Return compact JSON for imaging report"},
        {"parameters", {
            {"type","object"},
            {"properties", {
                {"patient_name", {{"type","string"}}},
                {"study_type", {{"type","string"}}},
                {"study_date", {{"type","string"}}},
                {"impression", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"findings", {{"type","array"}, {"items", {{"type","string"}}}}},
                {"confidence", {{"type","number"}}}
            }},
            {"required", json::array({"impression","confidence"})}
        }}
    };
}

static json build_functions_for(DocType dt) {
    switch(dt) {
        case DocType::MEDICAL: return json::array({schema_medical()});
        case DocType::PLEADING: return json::array({schema_pleading()});
        case DocType::POLICE: return json::array({schema_police()});
        case DocType::TRANSCRIPT: return json::array({schema_transcript()});
        case DocType::INSURANCE_EOB: return json::array({schema_eob()});
        case DocType::IMAGING: return json::array({schema_imaging()});
        default: return json::array({schema_medical(), schema_pleading(), schema_police(), schema_transcript(), schema_eob(), schema_imaging()});
    }
}
static std::string func_name_for(DocType dt) {
    switch(dt) {
        case DocType::MEDICAL: return "extract_medical_json";
        case DocType::PLEADING: return "extract_pleading_json";
        case DocType::POLICE: return "extract_police_json";
        case DocType::TRANSCRIPT: return "extract_transcript_json";
        case DocType::INSURANCE_EOB: return "extract_eob_json";
        case DocType::IMAGING: return "extract_imaging_json";
        default: return "extract_medical_json";
    }
}

// ---------------- Rate limit and backoff ----------------
struct RateLimiter {
    std::mutex mu;
    std::chrono::steady_clock::time_point next_ok = std::chrono::steady_clock::now();
    int qps = 3; // rough client side limit
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        auto now = std::chrono::steady_clock::now();
        if (now < next_ok) std::this_thread::sleep_until(next_ok);
        next_ok = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000 / std::max(1,qps));
    }
} limiter;

// ---------------- OpenAI call ----------------
static json call_openai_compact(const Config &cfg, DocType dt, const json &local_candidates, const std::string &snippet) {
    json req;
    req["model"] = cfg.model;
    req["temperature"] = 0.0;

    json messages = json::array();
    messages.push_back({
        {"role","system"},
        {"content","You extract structured data for legal and medical workflows. Return only compact JSON matching the function schema, no extra text."}
    });

    json u = {
        {"role","user"},
        {"content",
            std::string("Document type guess: ") + doc_type_str(dt) +
            ". Keep output minified JSON only.\n" +
            local_candidates.dump() + "\n---\n" +
            snippet.substr(0, cfg.max_chars_per_snippet)
        }
    };
    messages.push_back(u);

    req["messages"] = messages;
    req["functions"] = build_functions_for(dt);
    req["function_call"] = { {"name", func_name_for(dt)} };

    long http_code = 0;
    json resp;
    int attempts = 0;
    int max_attempts = 4;
    int backoff_ms = 400;
    while (attempts < max_attempts) {
        limiter.wait();
        resp = http_post_json("https://api.openai.com/v1/chat/completions", cfg.api_key, req, http_code, cfg.http_timeout);
        if (http_code >= 500) {
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms *= 2;
            attempts++;
            continue;
        }
        if (http_code == 429) {
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms = std::min(5000, backoff_ms * 2);
            attempts++;
            continue;
        }
        break;
    }
    if (http_code >= 400) {
        std::cerr << "OpenAI HTTP " << http_code << ": " << resp.dump() << std::endl;
        die("OpenAI request failed");
    }

    // parse function_call.arguments or content, with basic repair if needed
    try {
        auto &choice = resp["choices"][0];
        std::string payload;
        if (choice.contains("message") && choice["message"].contains("function_call")) {
            payload = choice["message"]["function_call"]["arguments"].get<std::string>();
        } else if (choice.contains("message") && choice["message"].contains("content")) {
            payload = choice["message"]["content"].get<std::string>();
        }
        // try parse, with brace repair
        try {
            return json::parse(payload);
        } catch (...) {
            auto start = payload.find('{');
            auto end = payload.rfind('}');
            if (start != std::string::npos && end != std::string::npos && end > start) {
                return json::parse(payload.substr(start, end - start + 1));
            }
            throw;
        }
    } catch (...) {
        std::cerr << "Raw response: " << resp.dump(2) << std::endl;
        die("Failed to parse model output JSON");
    }
    return {};
}

// ---------------- Merge and redact ----------------
static json merge_local_and_model(DocType dt, const json &local_cand, json model) {
    if (!model.contains("snippets") && local_cand.contains("important_snippets")) {
        model["snippets"] = local_cand["important_snippets"];
    }
    if (local_cand.contains("name_candidate")) {
        if (!model.contains("patient_name")) model["patient_name"] = local_cand["name_candidate"];
        if (!model.contains("member")) model["member"] = local_cand["name_candidate"];
    }
    if (dt == DocType::TRANSCRIPT && local_cand.contains("local_citations")) {
        // if model has no citations, add locals
        if (!model.contains("citations")) model["citations"] = local_cand["local_citations"];
    }
    return model;
}

static void redact_in_place(json &j) {
    // simple redactions for phone, SSN, email
    auto redact_string = [](std::string s){
        s = std::regex_replace(s, std::regex(R"((\b\d{3}[- ]?\d{2}[- ]?\d{4}\b))"), "***-**-****");
        s = std::regex_replace(s, std::regex(R"((\+?\d{1,2}[\s\-\.])?(?:\(?\d{3}\)?[\s\-\.])?\d{3}[\s\-\.]\d{4})"), "***-***-****");
        s = std::regex_replace(s, std::regex(R"(([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}))"), "***@***.***");
        return s;
    };
    std::function<void(json&)> walk = [&](json &node){
        if (node.is_string()) {
            std::string val = node.get<std::string>();
            node = redact_string(val);
        } else if (node.is_array()) {
            for (auto &el : node) walk(el);
        } else if (node.is_object()) {
            for (auto &kv : node.items()) walk(kv.value());
        }
    };
    walk(j);
}

// ---------------- Cache ----------------
static bool cache_load(const Config &cfg, const std::string &key, json &out) {
    if (cfg.cache_dir.empty()) return false;
    fs::create_directories(cfg.cache_dir);
    std::string path = (fs::path(cfg.cache_dir) / (key + ".json")).string();
    std::ifstream f(path);
    if (!f) return false;
    try {
        out = json::parse(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
        return true;
    } catch (...) { return false; }
}
static void cache_store(const Config &cfg, const std::string &key, const json &val) {
    if (cfg.cache_dir.empty()) return;
    fs::create_directories(cfg.cache_dir);
    std::string path = (fs::path(cfg.cache_dir) / (key + ".json")).string();
    std::ofstream f(path);
    if (f) f << val.dump();
}

// ---------------- Document processing ----------------
struct DocResult {
    std::string input_path;
    DocType doc_type = DocType::UNKNOWN;
    json result_json;
    bool ok = false;
    std::string error;
    int pages = 0;
    int chars_used = 0;
};

static std::string concat_for_selection(const std::vector<std::string> &page_texts, size_t max_lines) {
    std::vector<std::string> lines;
    for (auto &t : page_texts) {
        std::istringstream iss(t);
        std::string line;
        while (std::getline(iss, line)) {
            line = trim_copy(line);
            if (!line.empty()) lines.push_back(line);
            if (lines.size() >= max_lines * 2) break;
        }
        if (lines.size() >= max_lines * 2) break;
    }
    std::string out;
    for (auto &l : lines) {
        if (out.size() + l.size() + 1 > 4000) break;
        out += l; out += "\n";
    }
    return out;
}

static DocResult process_single_document(const fs::path &path, const Config &cfg) {
    DocResult r;
    r.input_path = path.string();

    try {
        std::vector<std::string> images;
        std::vector<std::string> page_texts;

        if (is_pdf(path)) {
            std::string tmpdir = (fs::temp_directory_path() / (path.stem().string() + "_ppm")).string();
            images = pdf_to_images(path.string(), tmpdir);
            if (images.empty()) die("No pages produced from " + path.string());
        } else if (is_image(path)) {
            images.push_back(path.string());
        } else {
            die("Unsupported file type: " + path.string());
        }

        for (auto &img : images) {
            std::string text = ocr_image_path(img, cfg);
            if (!text.empty()) page_texts.push_back(text);
        }
        if (page_texts.empty()) die("OCR produced no text for " + path.string());
        r.pages = (int)images.size();

        std::string full_concat;
        for (auto &t : page_texts) {
            full_concat += t;
            if (full_concat.size() > 40000) break;
        }

        DocType dt = classify_doc(full_concat);
        r.doc_type = dt;

        std::string selection = concat_for_selection(page_texts, cfg.max_snippet_lines);
        json local = local_extract_by_type(selection.empty() ? page_texts.front() : selection, dt, cfg);

        // Build snippet key for cache
        std::string cache_material = std::string(doc_type_str(dt)) + "\n" + local.dump();
        uint64_t h = fnv1a_64(cache_material);
        std::string key = std::to_string(h);

        json model;
        if (!cache_load(cfg, key, model)) {
            model = call_openai_compact(cfg, dt, local, local.value("important_snippets",""));
            cache_store(cfg, key, model);
        }

        json merged = merge_local_and_model(dt, local, model);
        merged["doc_type"] = doc_type_str(dt);
        merged["source"] = path.filename().string();
        merged["page_count"] = r.pages;
        if (cfg.audit_raw_ocr) {
            // keep only the first 4000 chars to avoid giant outputs
            std::string raw = full_concat.substr(0, 4000);
            merged["raw_ocr_preview"] = raw;
        }

        if (cfg.redact) redact_in_place(merged);

        r.chars_used = (int)local.value("important_snippets", "").size();
        r.result_json = merged;
        r.ok = true;
    } catch (const std::exception &e) {
        r.ok = false;
        r.error = e.what();
    } catch (...) {
        r.ok = false;
        r.error = "unknown error";
    }
    return r;
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    Config cfg = parse_cli(argc, argv);
    curl_global_init(CURL_GLOBAL_ALL);

    std::vector<fs::path> inputs;
    if (fs::is_directory(cfg.input_path)) {
        for (auto &entry : fs::directory_iterator(cfg.input_path)) {
            if (!entry.is_regular_file()) continue;
            if (is_pdf(entry.path()) || is_image(entry.path())) inputs.push_back(entry.path());
        }
        if (inputs.empty()) die("No PDFs or images found in folder");
        std::sort(inputs.begin(), inputs.end());
    } else {
        inputs.push_back(cfg.input_path);
    }

    std::mutex io_mu;
    std::vector<DocResult> results(inputs.size());
    std::atomic<size_t> idx{0};

    int thread_count = std::min<int>(cfg.threads, (int)inputs.size());
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    // optional JSONL
    std::unique_ptr<std::ofstream> jsonl_stream;
    if (!cfg.jsonl_path.empty()) {
        jsonl_stream.reset(new std::ofstream(cfg.jsonl_path, std::ios::out));
        if (!*jsonl_stream) die("Cannot open jsonl path");
    }

    auto write_per_file = [&](const DocResult &r){
        if (!cfg.per_file || !r.ok) return;
        fs::path p = r.input_path;
        fs::path outp = p.parent_path() / (p.stem().string() + ".extracted.json");
        std::ofstream f(outp);
        if (f) f << r.result_json.dump();
    };

    auto write_jsonl = [&](const DocResult &r){
        if (!jsonl_stream || !*jsonl_stream) return;
        json one;
        one["ok"] = r.ok;
        one["source"] = r.input_path;
        one["doc_type"] = doc_type_str(r.doc_type);
        one["page_count"] = r.pages;
        if (r.ok) one["data"] = r.result_json;
        else one["error"] = r.error;
        (*jsonl_stream) << one.dump() << "\n";
        jsonl_stream->flush();
    };

    auto worker = [&](){
        while (true) {
            size_t i = idx.fetch_add(1);
            if (i >= inputs.size()) break;
            DocResult r = process_single_document(inputs[i], cfg);
            {
                std::lock_guard<std::mutex> lk(io_mu);
                results[i] = std::move(r);
                write_per_file(results[i]);
                write_jsonl(results[i]);
                std::cout << "[" << i+1 << "/" << inputs.size() << "] "
                          << fs::path(results[i].input_path).filename().string()
                          << " -> " << (results[i].ok ? "OK" : "ERR") << "\n";
            }
        }
    };

    for (int t = 0; t < thread_count; ++t) workers.emplace_back(worker);
    for (auto &th : workers) th.join();

    // Combined JSON
    json out;
    out["generated_at"] = (long long)std::time(nullptr);
    out["model"] = cfg.model;
    out["documents"] = json::array();
    out["errors"] = json::array();

    size_t total_chars = 0;
    for (auto &r : results) {
        if (r.ok) {
            out["documents"].push_back(r.result_json);
            total_chars += r.chars_used;
        } else {
            out["errors"].push_back({{"source", r.input_path}, {"error", r.error}});
        }
    }
    out["stats"] = {
        {"processed", results.size()},
        {"ok", out["documents"].size()},
        {"errors", out["errors"].size()},
        {"avg_snippet_chars", out["documents"].size() ? (int)(total_chars / std::max<size_t>(1, out["documents"].size())) : 0}
    };

    std::ofstream f(cfg.output_json);
    if (!f) die("Failed to open output file");
    f << out.dump();
    f.close();

    if (jsonl_stream && *jsonl_stream) {
        std::cout << "JSONL written: " << cfg.jsonl_path << "\n";
    }
    std::cout << "Combined JSON written: " << cfg.output_json << "\n";

    curl_global_cleanup();
    return 0;
}
