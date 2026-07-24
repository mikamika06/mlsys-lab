#pragma once
#include <string>

// A self-contained, real-C++ model of the CPython C-API error-propagation
// protocol: a per-call "error indicator" you set with an analogue of
// PyErr_SetString, plus a PyObject*-style return value where nullptr means
// "an error is set, check it" and non-null means "success, no error is set".
// All of this (ExcState, g_exc, set_error, PyFloatObj) is harness code,
// defined in main.cpp — you only implement safe_divide.

enum class ExcType { None, ZeroDivisionError };

// Analogue of CPython's per-thread error indicator.
struct ExcState {
    ExcType type = ExcType::None;
    std::string message;
};
extern ExcState g_exc;   // reset to {ExcType::None, ""} by the driver before every call

// Analogue of PyErr_SetString(type, msg): records the exception in g_exc.
void set_error(ExcType type, const std::string& msg);

// Analogue of a PyObject* wrapping a Python float (PyFloat_FromDouble).
struct PyFloatObj {
    double value;
};

// Analogue of a CPython C-API wrapper function `safe_divide(a, b)`:
//
//   - if b == 0.0: call set_error(ExcType::ZeroDivisionError,
//     "division by zero") and return nullptr. NEVER return a non-null
//     object after setting an error — that is exactly the CPython bug this
//     task is about.
//   - otherwise: leave g_exc untouched (ExcType::None) and return a freshly
//     allocated `new PyFloatObj{a / b}`. NEVER return nullptr without an
//     error set — the caller would treat that as "no error, no result" and
//     crash on the null dereference.
PyFloatObj* safe_divide(double a, double b);
