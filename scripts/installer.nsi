!include "MUI2.nsh"
!include "FileFunc.nsh"

; Define application name and version from command line
!ifndef VERSION
    !define VERSION "0.0.1-beta"
!endif

!ifndef STAGING_DIR
    !define STAGING_DIR "staging"
!endif

; Application name and version
Name "AV LLM ${VERSION}"
OutFile "av_llm-windows-installer-${VERSION}.exe"

; Default installation directory
InstallDir "$PROGRAMFILES\AV LLM"

; Request admin privileges
RequestExecutionLevel admin

; Interface Settings
!define MUI_ABORTWARNING

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    
    ; Copy binary
    SetOutPath "$INSTDIR\bin"
    File "${STAGING_DIR}\bin\av_llm.exe"
    
    ; Copy model and configuration
    SetOutPath "$INSTDIR\etc\.av_llm"
    File "${STAGING_DIR}\etc\.av_llm\qwen1_5-0_5b-chat-q8_0.gguf"
    
    ; Add to PATH
    ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "$INSTDIR\bin"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"
    
    ; Create start menu shortcuts
    CreateDirectory "$SMPROGRAMS\AV LLM"
    CreateShortcut "$SMPROGRAMS\AV LLM\AV LLM.lnk" "$INSTDIR\bin\av_llm.exe"
    CreateShortcut "$SMPROGRAMS\AV LLM\Uninstall.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
    ; Remove files
    RMDir /r "$INSTDIR\bin"
    RMDir /r "$INSTDIR\etc"
    Delete "$INSTDIR\uninstall.exe"
    RMDir "$INSTDIR"
    
    ; Remove start menu shortcuts
    RMDir /r "$SMPROGRAMS\AV LLM"
    
    ; Remove from PATH
    ${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" "$INSTDIR\bin"
SectionEnd
