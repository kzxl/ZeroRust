@echo off
setlocal
echo Compiling ZeroCompute C Sample...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl.exe /O2 /W4 /I..\..\include main.c ..\..\target\release\zero_compute.dll.lib /Fe:main.exe
if %ERRORLEVEL% EQU 0 (
    copy /Y ..\..\target\release\zero_compute.dll .\ >nul
    echo Running main.exe...
    main.exe
) else (
    echo Compilation failed.
)
