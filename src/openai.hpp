#ifndef _OPENAI_API_
#define _OPENAI_API_

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace av_llm::openai {

struct Model
{
    std::string id;
    std::string object;
    int64_t created;
    std::string owned_by;

    // Default constructor
    Model() : id(""), object("model"), created(0), owned_by("") {}

    // Parameterized constructor
    Model(const std::string & id, const std::string & object, int64_t created, const std::string & owned_by) :
        id(id), object(object), created(created), owned_by(owned_by)
    {}

    // Constructor from JSON
    Model(const json & j)
    {
        id       = j.at("id").get<std::string>();
        object   = j.at("object").get<std::string>();
        created  = j.at("created").get<int64_t>();
        owned_by = j.at("owned_by").get<std::string>();
    }

    // Convert to JSON
    json to_json() const { return json{ { "id", id }, { "object", object }, { "created", created }, { "owned_by", owned_by } }; }
};

// Struct to represent the entire model list
struct ModelList
{
    std::string object;
    std::vector<Model> data;

    // Default constructor
    ModelList() : object("list") {}

    // Constructor with data
    ModelList(const std::vector<Model> & models) : object("list"), data(models) {}

    // Constructor from JSON
    ModelList(const json & j)
    {
        object = j.at("object").get<std::string>();

        // Parse the data array
        for (const auto & model_json : j.at("data"))
        {
            data.emplace_back(Model(model_json));
        }
    }

    // Convert to JSON
    json to_json() const
    {
        json result;
        result["object"] = object;

        // Convert each model to JSON
        json data_array = json::array();
        for (const auto & model : data)
        {
            data_array.push_back(model.to_json());
        }
        result["data"] = data_array;

        return result;
    }

    // Add a model to the list
    void add_model(const Model & model) { data.push_back(model); }

    // Get number of models
    size_t size() const { return data.size(); }
};
// Convenience functions for nlohmann/json automatic serialization
void to_json(json & j, const Model & m)
{
    j = m.to_json();
}

void from_json(const json & j, Model & m)
{
    m = av_llm::openai::Model(j);
}

void to_json(json & j, const ModelList & ml)
{
    j = ml.to_json();
}

void from_json(const json & j, av_llm::openai::ModelList & ml)
{
    ml = av_llm::openai::ModelList(j);
}

} // namespace av_llm::openai
#endif
