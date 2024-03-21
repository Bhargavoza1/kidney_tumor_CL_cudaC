 
 
#include <vector>
 
#include <sstream>
 

namespace Hex {

    template<typename T>
    std::string shapeToString(const std::vector<T>& shape) {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i];
            if (i < shape.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    
}
