#pragma once

struct Plot128Context;

class Plot128Phase1
{
public:
    Plot128Phase1( Plot128Context& cx );

    void Run();
private:
    Plot128Context& _cx;
};