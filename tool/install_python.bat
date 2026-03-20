@echo off
setlocal enabledelayedexpansion

echo ==============================
echo   Python Auto Installer PRO
echo ==============================

:: Nhập phiên bản Python
set /p pyver=Nhap version Python (vd: 3.10.9):

:: Link download
set url=https://www.python.org/ftp/python/%pyver%/python-%pyver%-amd64.exe
set filename=python-%pyver%-amd64.exe

echo.
echo Installing Python %pyver%...
powershell -Command "Invoke-WebRequest -Uri '%url%' -OutFile '%filename%'"

if not exist "%filename%" (
    echo Loi: Khong tai duoc file. Version co the khong ton tai!
    pause
    exit /b
)

echo.
set /p addpath=Ban co muon add Python vao PATH khong? (y/n):

if /i "%addpath%"=="y" (
    echo Cai dat voi PATH...
    %filename% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
) else (
    echo Cai dat KHONG them PATH...
    %filename% /quiet InstallAllUsers=1 PrependPath=0 Include_test=0
)

echo.
echo Dang kiem tra phien ban Python...

:: Đợi vài giây cho hệ thống cập nhật PATH
timeout /t 3 >nul

:: Lấy version từ command
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    set installed_ver=%%v
)

:: Nếu không lấy được version
if not defined installed_ver (
    echo.
    echo [LOI] Khong tim thay lenh python!
    echo Co the ban chua add PATH hoac can mo lai CMD.
    pause
    exit /b
)

echo Version da cai: !installed_ver!
echo Version yeu cau: %pyver%

:: So sánh version
if "!installed_ver!"=="%pyver%" (
    echo.
    echo [OK] Cai dat thanh cong dung version!
) else (
    echo.
    echo [LOI] Version KHONG KHOP!
    echo Co the:
    echo - PATH dang tro den Python cu
    echo - Cai dat bi loi
)

:: Xoa file cai dat
del "%filename%"

echo.
pause