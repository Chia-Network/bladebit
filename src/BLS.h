#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

#pragma warning( push )

extern "C" {
    #include "bech32/segwit_addr.h"
}

#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma warning( disable : 6287  )
#pragma warning( disable : 4267  )
#pragma warning( disable : 26495 )
#include "bls.hpp"
#include "elements.hpp"
#include "schemes.hpp"
#include "util.hpp"
#pragma GCC diagnostic pop
#pragma warning( pop )