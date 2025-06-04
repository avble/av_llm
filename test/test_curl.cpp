#include "catch2/catch.hpp"

#include <curl/curl.h>
#include <iostream>
#include <variant>

static size_t write_data(void * ptr, size_t size, size_t nmemb, void * stream)
{
    return fwrite(ptr, size, nmemb, (FILE *) stream);
}

// Progress callback (older interface)
int progress_callback(void * /*clientp*/, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
    if (dltotal == 0)
        return 0; // avoid division by zero

    double progress = (double) dlnow / (double) dltotal * 100.0;
    std::cout << "\rDownload progress: " << progress << "% (" << dlnow << "/" << dltotal << " bytes)" << std::flush;

    return 0; // return non-zero to abort transfer
}

TEST_CASE("test_curl_01")
{

    if (true)
    {

        static auto get_file_name_from_url = [](const std::string url) -> std::string {
            auto last_slash = url.find_last_of('/');

            if (last_slash == std::string::npos)
                return url;

            return url.substr(last_slash + 1);
        };

        // struct hugging_face_t
        // {
        //     // input:
        //     // https://huggingface.co/ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF/resolve/main/qwen2.5-coder-0.5b-q8_0.gguf

        //     hugging_face_t(std::string _user, std::string _model) { url = _user + ":" + _model; }

        //     // std::string operator=(const hugging_face_t & other) { return other.user + other.model; }
        //     // const char * c_str() { return url.c_str(); }
        //     std::string get_url() { return url; }
        //     std::string get_file_name() { return get_file_name_from_url(url); }

        //     std::string url;
        // };

        // struct url_from_string_t : public std::string
        // {

        //     url_from_string_t(std::string _url) : std::string(_url) {}

        //     std::string get_url() { return std::string(this->c_str()); }
        //     std::string get_file_name() { return get_file_name_from_url(*this); }
        // };

        // static auto download_model = [](std::variant<std::string, hugging_face_t> from) {
        //     std::string url         = std::visit([](auto & obj) -> std::string { return obj.get_url(); }, from);
        //     std::string outfilename = std::visit([](auto & obj) -> std::string { return obj.get_file_name(); }, from);

        static auto download_model = [](std::string url) {
            CURL * curl;
            FILE * fp;
            CURLcode res;

            std::string outfilename = get_file_name_from_url(url);

            curl_global_init(CURL_GLOBAL_DEFAULT);
            curl = curl_easy_init();
            if (curl)
            {
                fp = fopen(outfilename.c_str(), "wb");
                if (fp)
                {
                    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

                    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

                    // Set progress callback
                    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
                    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

                    // follow redirect
                    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

                    res = curl_easy_perform(curl);
                    if (res != CURLE_OK)
                        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                    fclose(fp);
                }

                curl_easy_cleanup(curl);
            }
            curl_global_cleanup();
        };

        std::string url = "https://example.com/file.zip";
        std::cout << "download: " << url << std::endl;
        download_model(url);
        url = "https://huggingface.co/ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF/resolve/main/qwen2.5-coder-3b-q8_0.gguf";
        std::cout << "download: " << url << std::endl;
        download_model(url);
    }

    if (false)
    {
        CURL * curl;
        FILE * fp;
        CURLcode res;
        const char * url         = "https://example.com/file.zip";
        const char * outfilename = "downloaded_file.zip";

        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl = curl_easy_init();
        if (curl)
        {
            fp = fopen(outfilename, "wb");
            if (fp)
            {
                curl_easy_setopt(curl, CURLOPT_URL, url);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

                // Set progress callback
                curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
                curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

                // follow redirect
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

                res = curl_easy_perform(curl);
                if (res != CURLE_OK)
                    fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                fclose(fp);
            }

            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }
}
