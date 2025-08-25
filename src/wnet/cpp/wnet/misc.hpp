#ifndef WNET_MISC_HPP
#define WNET_MISC_HPP


inline std::string get_type_str(const nb::object &obj) {
    nb::handle type = obj.type();
    nb::str name = nb::str(type.attr("__name__"));
    nb::str module = nb::str(type.attr("__module__"));
    return std::string(module.c_str()) + "." + std::string(name.c_str());
}


#endif // WNET_MISC_HPP