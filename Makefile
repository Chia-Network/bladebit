# Usage: make -j<#threads> CONFIG=<config_name>

# --------------------------------------------------------
# Public Config
# --------------------------------------------------------
CC			:= gcc
CXX			:= g++
AR			:= ar

PLATFORM	:= linux
CONFIG		:= release
VERBOSE		:= 0

OBJ_DIR		:= .obj
OBJ_EXT		:= o
OUT_DIR		:= ./bin
OUT_NAME	:= bladebit
Q			:= @

PCH 		:= src/pch.h

is_release	:=
is_arm		:=

# Detect platform
ifeq ($(OS),Windows_NT)
	PLATFORM := win32
else
	UNAME := $(shell uname -s)

	ifeq ($(UNAME),Linux)
		PLATFORM := linux
	endif

	ifeq ($(UNAME),Darwin)
		PLATFORM := macos
	endif
endif

LDFLAGS := -pthread \
	-Llib \
	-lbls \
	-lnuma


# Is it release or debug?
ifneq (,$(findstring release,$(CONFIG)))
	is_release := 1
endif

ifneq (,$(findstring arm,$(CONFIG)))
	is_arm := 1
endif

# Platform-specific configs
ifeq ($(PLATFORM),macos)
	CC  := clang
	CXX := clang++

	LDFLAGS += -Llib/macos
endif

ifeq ($(PLATFORM),linux)
	ifeq ($(is_arm),1)
		LDFLAGS += -Llib/linux/arm
	else
		LDFLAGS += -Llib/linux/x86
	endif
endif

CFLAGS := \
	-Wall \
	-Wno-comment

CXXFLAGS    := 	\
	-std=c++17
	

# GCC or Clang?
ifneq (,$(findstring gcc,$(CC)))
	CFLAGS += \
		-fmax-errors=5
endif

ifneq (,$(findstring clang,$(CC)))
	CFLAGS += \
		-ferror-limit=5 \
		-fdeclspec 		\
		-fno-exceptions \
		-Wunknown-pragmas
endif

# For ARM
ifeq ($(is_arm),1)
# CFLAGS += -mfpu=neon-vfpv4
else
endif



# Include precompiled header in all compilation units
ifneq ($(PCH),)
	CFLAGS += --include=$(PCH)
endif

# --------------------------------------------------------
# Private Config
# --------------------------------------------------------
SOURCES 	:=
SRC_ROOT	:= src
OPT_FLAGS	:=


ifeq ($(is_release),1)
	OPT_FLAGS := -O3 -flto -g
else
	OPT_FLAGS := -O0 -g
endif

CFLAGS += $(OPT_FLAGS)

ifneq ($(VERBOSE),0)
	Q := 
endif


# Include config vars
include builds/$(PLATFORM)/$(CONFIG).mk

#
# prepare sources and objects
#

sources = $(SOURCES)

objects := $(sources:$(SRC_ROOT)/%=$(OBJ_DIR)/%)
objects := $(objects:.cpp=.$(OBJ_EXT))
objects := $(objects:.c=.$(OBJ_EXT))
objects := $(objects:.S=.$(OBJ_EXT))

# --------------------------------------------------------
# Targets
# --------------------------------------------------------
.PHONY: clean compile bladebit lib

default: bladebit

# static library
lib: compile
	@mkdir -p $(OUT_DIR)/$(CONFIG)
	$(Q)$(AR) rc $(OUT_DIR)/$(CONFIG)/lib$(OUT_NAME).a $(objects)


# make exe
bladebit: compile
	@mkdir -p $(OUT_DIR)/$(CONFIG)
	$(Q)$(CXX) $(OPT_FLAGS) $(objects) -o $(OUT_DIR)/$(CONFIG)/$(OUT_NAME) $(LDFLAGS)
	@objcopy --only-keep-debug $(OUT_DIR)/$(CONFIG)/$(OUT_NAME) $(OUT_DIR)/$(CONFIG)/$(OUT_NAME).debug
	@strip --strip-debug --strip-unneeded $(OUT_DIR)/$(CONFIG)/$(OUT_NAME)
	@objcopy --add-gnu-debuglink="$(OUT_DIR)/$(CONFIG)/$(OUT_NAME).debug" $(OUT_DIR)/$(CONFIG)/$(OUT_NAME)

# compile sources
compile: info $(objects)

info:
	@echo "Compiling for $(PLATFORM) with $(CXX)"
# @echo $(objects)
clean:
	@rm -rf $(OBJ_DIR)
	@rm -rf $(OUT_DIR)

# Compile ASM
$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_ROOT)/%.S
	@echo $<
	@mkdir -p $(@D)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@
	$(Q)$(CC) $(CFLAGS) -c $< -MM -MT $(@D)/$(*F).o -o $(@D)/$(*F).d

# Compile C
$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_ROOT)/%.c
	@echo $<
	@mkdir -p $(@D)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@
	$(Q)$(CC) $(CFLAGS) -c $< -MM -MT $(@D)/$(*F).o -o $(@D)/$(*F).d

# Compile C++
$(OBJ_DIR)/%.$(OBJ_EXT): $(SRC_ROOT)/%.cpp 
	@echo $<
	@mkdir -p $(@D)
	$(Q)$(CXX) $(CFLAGS) $(CXXFLAGS) -c $< -o $@
	$(Q)$(CXX) $(CFLAGS) $(CXXFLAGS) -c $< -MM -MT $(@D)/$(*F).o -o $(@D)/$(*F).d

