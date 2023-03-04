#pragma once

#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"

void CmdSimulateHelp();
void CmdSimulateMain( GlobalPlotConfig& gCfg, CliParser& cli );
