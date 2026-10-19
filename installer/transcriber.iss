; One-click installer. Build it with installer\build.ps1.
;
; Per-user, so no admin prompt. Every wizard page is off, so opening setup starts the
; install. After the files are copied, [Code] runs three steps: deps.cmd downloads Python
; and the dependencies, `transcriber.exe --setup` tests the hardware and picks and
; downloads the models, then the app starts.

#define AppName "Transcriber"
#define AppVersion "0.1.0"
#define AppExe "{app}\.venv\Scripts\transcriber.exe"

[Setup]
AppId={{8C4E2B7A-3F1D-4E6B-9A2C-5D7E1F0B3A94}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=aarav
; ponytail: keep this path short. The uninstaller is bound by MAX_PATH (260), and the uv
; cache holds 193-character paths: a 20-character username reaches 258. Move uv-cache out,
; or delete it after sync, if a deeper dependency ever pushes past that.
DefaultDirName={localappdata}\Programs\{#AppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableFinishedPage=yes
; The app holds this mutex (app/main.py), so setup and uninstall ask for it to be closed.
AppMutex=TranscriberRunning
UninstallDisplayName={#AppName}
UninstallDisplayIcon={#AppExe}
OutputDir=..\dist
OutputBaseFilename=TranscriberSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "..\app\*"; DestDir: "{app}\app"; Excludes: "__pycache__"; Flags: recursesubdirs ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uv.lock"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.python-version"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "deps.cmd"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{#AppExe}"; WorkingDir: "{app}"

[Registry]
; Only removed here, at uninstall. `--setup` writes the start-at-login entry through
; app/autostart.py: security software was seen denying this unsigned setup that write
; (RegSetValueEx code 5) while allowing it from Python. noerror: a denial must not stop
; the uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; \
  ValueName: "Transcriber"; Flags: uninsdeletevalue noerror

[UninstallDelete]
; Made after install, so uninstall does not know about them otherwise. History and
; settings in %APPDATA%\transcriber are the user's and stay.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\uv-cache"
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{app}\install.log"

[Code]
function Step(const Caption, FileName, Params: String; Show: Integer): Integer;
begin
  WizardForm.StatusLabel.Caption := Caption;
  WizardForm.FilenameLabel.Caption := '';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  if not Exec(FileName, Params, ExpandConstant('{app}'), Show, ewWaitUntilTerminated, Result) then
    Result := -1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Extra: String;
  Code: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;

  { The CUDA wheels are 2 GB that only an NVIDIA GPU can use. }
  Extra := '';
  if FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
    Extra := ' --extra cuda';

  if Step('Downloading Python and components (up to 3 GB). This can take several minutes...',
          ExpandConstant('{cmd}'),
          '/C ""' + ExpandConstant('{app}\deps.cmd') + '"' + Extra + ' > "' +
            ExpandConstant('{app}\install.log') + '" 2>&1"',
          SW_HIDE) <> 0 then
  begin
    { Suppressible, so a /SUPPRESSMSGBOXES install cannot hang on it. }
    SuppressibleMsgBox('Setup could not install its components. Check your internet ' +
           'connection, and that security software did not block setup, then run setup ' +
           'again.' #13#10#13#10 'Details: ' + ExpandConstant('{app}\install.log'),
           mbCriticalError, MB_OK, IDOK);
    Exit;
  end;

  { Not fatal: on failure the app starts with its defaults, and the log says why. }
  Step('Testing your hardware and downloading models...', ExpandConstant('{#AppExe}'),
       '--setup', SW_SHOW);

  Exec(ExpandConstant('{#AppExe}'), '', ExpandConstant('{app}'), SW_SHOW, ewNoWait, Code);
end;
