#pragma once


struct ValidatePlotOptions
{
    std::string plotPath    = "";
    bool        inRAM       = false;
    uint32      threadCount = 0;
    float       startOffset = 0.0f; // Offset percent at which to start
};

