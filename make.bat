@echo off
REM Default target is "all" if none is provided
set TARGET=%1
if "%TARGET%"=="" set TARGET=all

REM Call nmake with Makefile.win and the chosen target
nmake /f Makefile.win %TARGET%
