// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/ObjectMacros.h"
#include "UObject/ScriptMacros.h"

PRAGMA_DISABLE_DEPRECATION_WARNINGS
#ifdef AIRSIM_ComputerVisionPawn_generated_h
#error "ComputerVisionPawn.generated.h already included, missing '#pragma once' in ComputerVisionPawn.h"
#endif
#define AIRSIM_ComputerVisionPawn_generated_h

#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_SPARSE_DATA
#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_RPC_WRAPPERS
#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_RPC_WRAPPERS_NO_PURE_DECLS
#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_INCLASS_NO_PURE_DECLS \
private: \
	static void StaticRegisterNativesAComputerVisionPawn(); \
	friend struct Z_Construct_UClass_AComputerVisionPawn_Statics; \
public: \
	DECLARE_CLASS(AComputerVisionPawn, APawn, COMPILED_IN_FLAGS(0 | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(AComputerVisionPawn)


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_INCLASS \
private: \
	static void StaticRegisterNativesAComputerVisionPawn(); \
	friend struct Z_Construct_UClass_AComputerVisionPawn_Statics; \
public: \
	DECLARE_CLASS(AComputerVisionPawn, APawn, COMPILED_IN_FLAGS(0 | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(AComputerVisionPawn)


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_STANDARD_CONSTRUCTORS \
	/** Standard constructor, called after all reflected properties have been initialized */ \
	NO_API AComputerVisionPawn(const FObjectInitializer& ObjectInitializer); \
	DEFINE_DEFAULT_OBJECT_INITIALIZER_CONSTRUCTOR_CALL(AComputerVisionPawn) \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, AComputerVisionPawn); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(AComputerVisionPawn); \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API AComputerVisionPawn(AComputerVisionPawn&&); \
	NO_API AComputerVisionPawn(const AComputerVisionPawn&); \
public:


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_ENHANCED_CONSTRUCTORS \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API AComputerVisionPawn(AComputerVisionPawn&&); \
	NO_API AComputerVisionPawn(const AComputerVisionPawn&); \
public: \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, AComputerVisionPawn); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(AComputerVisionPawn); \
	DEFINE_DEFAULT_CONSTRUCTOR_CALL(AComputerVisionPawn)


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_PRIVATE_PROPERTY_OFFSET \
	FORCEINLINE static uint32 __PPO__pip_camera_class_() { return STRUCT_OFFSET(AComputerVisionPawn, pip_camera_class_); } \
	FORCEINLINE static uint32 __PPO__camera_front_center_base_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_center_base_); } \
	FORCEINLINE static uint32 __PPO__camera_front_left_base_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_left_base_); } \
	FORCEINLINE static uint32 __PPO__camera_front_right_base_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_right_base_); } \
	FORCEINLINE static uint32 __PPO__camera_bottom_center_base_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_bottom_center_base_); } \
	FORCEINLINE static uint32 __PPO__camera_back_center_base_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_back_center_base_); } \
	FORCEINLINE static uint32 __PPO__camera_front_center_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_center_); } \
	FORCEINLINE static uint32 __PPO__camera_front_left_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_left_); } \
	FORCEINLINE static uint32 __PPO__camera_front_right_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_front_right_); } \
	FORCEINLINE static uint32 __PPO__camera_bottom_center_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_bottom_center_); } \
	FORCEINLINE static uint32 __PPO__camera_back_center_() { return STRUCT_OFFSET(AComputerVisionPawn, camera_back_center_); } \
	FORCEINLINE static uint32 __PPO__manual_pose_controller_() { return STRUCT_OFFSET(AComputerVisionPawn, manual_pose_controller_); }


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_18_PROLOG
#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_GENERATED_BODY_LEGACY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_RPC_WRAPPERS \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_INCLASS \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_STANDARD_CONSTRUCTORS \
public: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


#define Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_GENERATED_BODY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_RPC_WRAPPERS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_INCLASS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h_21_ENHANCED_CONSTRUCTORS \
private: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


template<> AIRSIM_API UClass* StaticClass<class AComputerVisionPawn>();

#undef CURRENT_FILE_ID
#define CURRENT_FILE_ID Blocks_Plugins_AirSim_Source_Vehicles_ComputerVision_ComputerVisionPawn_h


PRAGMA_ENABLE_DEPRECATION_WARNINGS
