/*
 * Key Idea:
 *
 *   Using alias declarations is easier to read than function pointers.
 */

#include <string>

// FP is a synonym for a pointer to a function taking an int and
// a const std::string& and returning nothing
typedef void (*FP)(int, const std::string&);    // typedef

// same meaning as above
// 使用别名定义类型而非typedef
using FP = void (*)(int, const std::string&);   // alias
                                                // declaration
