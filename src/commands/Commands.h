#pragma once

#include "plotting/GlobalPlotConfig.h"
#include "util/CliParser.h"

void CmdSimulateHelp();
void CmdSimulateMain( GlobalPlotConfig& gCfg, CliParser& cli );

void CmdPlotsCheckHelp();
void CmdPlotsCheckMain( GlobalPlotConfig& gCfg, CliParser& cli );