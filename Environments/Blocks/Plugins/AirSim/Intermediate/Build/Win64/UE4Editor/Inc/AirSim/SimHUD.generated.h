// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/ObjectMacros.h"
#include "UObject/ScriptMacros.h"

PRAGMA_DISABLE_DEPRECATION_WARNINGS
#ifdef AIRSIM_SimHUD_generated_h
#error "SimHUD.generated.h already included, missing '#pragma once' in SimHUD.h"
#endif
#define AIRSIM_SimHUD_generated_h

#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_SPARSE_DATA
#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_RPC_WRAPPERS
#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_RPC_WRAPPERS_NO_PURE_DECLS
#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_INCLASS_NO_PURE_DECLS \
private: \
	static void StaticRegisterNativesASimHUD(); \
	friend struct Z_Construct_UClass_ASimHUD_Statics; \
public: \
	DECLARE_CLASS(ASimHUD, AHUD, COMPILED_IN_FLAGS(0 | CLASS_Transient | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(ASimHUD)


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_INCLASS \
private: \
	static void StaticRegisterNativesASimHUD(); \
	friend struct Z_Construct_UClass_ASimHUD_Statics; \
public: \
	DECLARE_CLASS(ASimHUD, AHUD, COMPILED_IN_FLAGS(0 | CLASS_Transient | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(ASimHUD)


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_STANDARD_CONSTRUCTORS \
	/** Standard constructor, called after all reflected properties have been initialized */ \
	NO_API ASimHUD(const FObjectInitializer& ObjectInitializer); \
	DEFINE_DEFAULT_OBJECT_INITIALIZER_CONSTRUCTOR_CALL(ASimHUD) \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, ASimHUD); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(ASimHUD); \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API ASimHUD(ASimHUD&&); \
	NO_API ASimHUD(const ASimHUD&); \
public:


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_ENHANCED_CONSTRUCTORS \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API ASimHUD(ASimHUD&&); \
	NO_API ASimHUD(const ASimHUD&); \
public: \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, ASimHUD); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(ASimHUD); \
	DEFINE_DEFAULT_CONSTRUCTOR_CALL(ASimHUD)


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_PRIVATE_PROPERTY_OFFSET \
	FORCEINLINE static uint32 __PPO__widget_() { return STRUCT_OFFSET(ASimHUD, widget_); } \
	FORCEINLINE static uint32 __PPO__simmode_() { return STRUCT_OFFSET(ASimHUD, simmode_); }


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_19_PROLOG
#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_GENERATED_BODY_LEGACY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_RPC_WRAPPERS \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_INCLASS \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_STANDARD_CONSTRUCTORS \
public: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


#define Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_GENERATED_BODY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_RPC_WRAPPERS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_INCLASS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h_22_ENHANCED_CONSTRUCTORS \
private: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


template<> AIRSIM_API UClass* StaticClass<class ASimHUD>();

#undef CURRENT_FILE_ID
#define CURRENT_FILE_ID Blocks_Plugins_AirSim_Source_SimHUD_SimHUD_h


#define FOREACH_ENUM_ESIMULATORMODE(op) \
	op(ESimulatorMode::SIM_MODE_HIL) 

enum class ESimulatorMode : uint8;
template<> AIRSIM_API UEnum* StaticEnum<ESimulatorMode>();

PRAGMA_ENABLE_DEPRECATION_WARNINGS
