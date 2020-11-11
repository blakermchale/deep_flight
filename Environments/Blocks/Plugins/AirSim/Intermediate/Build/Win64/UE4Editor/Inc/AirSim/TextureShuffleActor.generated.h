// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/ObjectMacros.h"
#include "UObject/ScriptMacros.h"

PRAGMA_DISABLE_DEPRECATION_WARNINGS
#ifdef AIRSIM_TextureShuffleActor_generated_h
#error "TextureShuffleActor.generated.h already included, missing '#pragma once' in TextureShuffleActor.h"
#endif
#define AIRSIM_TextureShuffleActor_generated_h

#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_SPARSE_DATA
#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_RPC_WRAPPERS \
	virtual void SwapTexture_Implementation(int32 tex_id, int32 component_id, int32 material_id); \
 \
	DECLARE_FUNCTION(execSwapTexture) \
	{ \
		P_GET_PROPERTY(UIntProperty,Z_Param_tex_id); \
		P_GET_PROPERTY(UIntProperty,Z_Param_component_id); \
		P_GET_PROPERTY(UIntProperty,Z_Param_material_id); \
		P_FINISH; \
		P_NATIVE_BEGIN; \
		P_THIS->SwapTexture_Implementation(Z_Param_tex_id,Z_Param_component_id,Z_Param_material_id); \
		P_NATIVE_END; \
	}


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_RPC_WRAPPERS_NO_PURE_DECLS \
	virtual void SwapTexture_Implementation(int32 tex_id, int32 component_id, int32 material_id); \
 \
	DECLARE_FUNCTION(execSwapTexture) \
	{ \
		P_GET_PROPERTY(UIntProperty,Z_Param_tex_id); \
		P_GET_PROPERTY(UIntProperty,Z_Param_component_id); \
		P_GET_PROPERTY(UIntProperty,Z_Param_material_id); \
		P_FINISH; \
		P_NATIVE_BEGIN; \
		P_THIS->SwapTexture_Implementation(Z_Param_tex_id,Z_Param_component_id,Z_Param_material_id); \
		P_NATIVE_END; \
	}


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_EVENT_PARMS \
	struct TextureShuffleActor_eventSwapTexture_Parms \
	{ \
		int32 tex_id; \
		int32 component_id; \
		int32 material_id; \
	};


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_CALLBACK_WRAPPERS
#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_INCLASS_NO_PURE_DECLS \
private: \
	static void StaticRegisterNativesATextureShuffleActor(); \
	friend struct Z_Construct_UClass_ATextureShuffleActor_Statics; \
public: \
	DECLARE_CLASS(ATextureShuffleActor, AStaticMeshActor, COMPILED_IN_FLAGS(0 | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(ATextureShuffleActor)


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_INCLASS \
private: \
	static void StaticRegisterNativesATextureShuffleActor(); \
	friend struct Z_Construct_UClass_ATextureShuffleActor_Statics; \
public: \
	DECLARE_CLASS(ATextureShuffleActor, AStaticMeshActor, COMPILED_IN_FLAGS(0 | CLASS_Config), CASTCLASS_None, TEXT("/Script/AirSim"), NO_API) \
	DECLARE_SERIALIZER(ATextureShuffleActor)


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_STANDARD_CONSTRUCTORS \
	/** Standard constructor, called after all reflected properties have been initialized */ \
	NO_API ATextureShuffleActor(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get()); \
	DEFINE_DEFAULT_OBJECT_INITIALIZER_CONSTRUCTOR_CALL(ATextureShuffleActor) \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, ATextureShuffleActor); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(ATextureShuffleActor); \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API ATextureShuffleActor(ATextureShuffleActor&&); \
	NO_API ATextureShuffleActor(const ATextureShuffleActor&); \
public:


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_ENHANCED_CONSTRUCTORS \
	/** Standard constructor, called after all reflected properties have been initialized */ \
	NO_API ATextureShuffleActor(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get()) : Super(ObjectInitializer) { }; \
private: \
	/** Private move- and copy-constructors, should never be used */ \
	NO_API ATextureShuffleActor(ATextureShuffleActor&&); \
	NO_API ATextureShuffleActor(const ATextureShuffleActor&); \
public: \
	DECLARE_VTABLE_PTR_HELPER_CTOR(NO_API, ATextureShuffleActor); \
DEFINE_VTABLE_PTR_HELPER_CTOR_CALLER(ATextureShuffleActor); \
	DEFINE_DEFAULT_OBJECT_INITIALIZER_CONSTRUCTOR_CALL(ATextureShuffleActor)


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_PRIVATE_PROPERTY_OFFSET \
	FORCEINLINE static uint32 __PPO__DynamicMaterial() { return STRUCT_OFFSET(ATextureShuffleActor, DynamicMaterial); } \
	FORCEINLINE static uint32 __PPO__SwappableTextures() { return STRUCT_OFFSET(ATextureShuffleActor, SwappableTextures); } \
	FORCEINLINE static uint32 __PPO__DynamicMaterialInstances() { return STRUCT_OFFSET(ATextureShuffleActor, DynamicMaterialInstances); }


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_11_PROLOG \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_EVENT_PARMS


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_GENERATED_BODY_LEGACY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_RPC_WRAPPERS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_CALLBACK_WRAPPERS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_INCLASS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_STANDARD_CONSTRUCTORS \
public: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


#define Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_GENERATED_BODY \
PRAGMA_DISABLE_DEPRECATION_WARNINGS \
public: \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_PRIVATE_PROPERTY_OFFSET \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_SPARSE_DATA \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_RPC_WRAPPERS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_CALLBACK_WRAPPERS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_INCLASS_NO_PURE_DECLS \
	Blocks_Plugins_AirSim_Source_TextureShuffleActor_h_14_ENHANCED_CONSTRUCTORS \
private: \
PRAGMA_ENABLE_DEPRECATION_WARNINGS


template<> AIRSIM_API UClass* StaticClass<class ATextureShuffleActor>();

#undef CURRENT_FILE_ID
#define CURRENT_FILE_ID Blocks_Plugins_AirSim_Source_TextureShuffleActor_h


PRAGMA_ENABLE_DEPRECATION_WARNINGS
