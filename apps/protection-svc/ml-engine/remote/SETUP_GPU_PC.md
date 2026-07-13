# GPU PC 최초 설정 (한 번만)

이 문서의 명령은 **GPU가 있는 그 PC**에서, **관리자 권한 PowerShell**로 실행한다. 지금 개발 중인 PC(dev PC)가 아니다.

## 1. OpenSSH Server 활성화

```powershell
# 설치 확인 후 설치
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# 서비스 시작 + 부팅 시 자동 시작
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# 방화벽 규칙 (보통 설치 시 자동 생성되지만 확인)
if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
}
```

## 2. 이 PC의 IP 주소와 사용자명 확인

```powershell
ipconfig | findstr IPv4
whoami
```

`IPv4 주소`(예: `192.168.0.42`)와 `whoami` 결과(예: `DESKTOP-XXXX\username`, `\` 뒤쪽만 사용)를 적어둔다.

## 3. Python + (선택) CUDA 확인

```powershell
python --version
nvidia-smi   # GPU 인식되는지, 드라이버 버전 확인
```

`nvidia-smi`가 실행되고 GPU가 보이면 준비 완료. Python이 없으면 [python.org](https://www.python.org/downloads/)에서 3.10~3.12 설치 (설치 시 "Add to PATH" 체크).

## 4. dev PC의 SSH 공개키 등록

dev PC(지금 이 대화가 실행되는 PC)에서 만든 공개키를 이 PC에 등록해야 비밀번호 없이 접속된다.

**먼저 이 계정이 로컬 Administrators 그룹 소속인지 확인** — `net user <계정명>`의 "Local Group Memberships"로는 놓치기 쉬우니, 더 확실한 방법:

```powershell
whoami /groups | findstr /i "Administrators"
```

뭔가 출력되면(비어있지 않으면) 관리자 계정이다. **Windows OpenSSH는 관리자 계정과 일반 계정을 완전히 다른 파일에서 키를 찾는다** — 이걸 헷갈리면 권한 설정을 다 맞게 해도 계속 거부당한다.

**일반 계정인 경우**, `%USERPROFILE%\.ssh\authorized_keys`:
```powershell
$key = "<dev PC의 공개키 한 줄>"
$authKeys = "$env:USERPROFILE\.ssh\authorized_keys"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null
[System.IO.File]::WriteAllText($authKeys, "$key`n", [System.Text.Encoding]::ASCII)
icacls $authKeys /inheritance:r
icacls $authKeys /grant "$($env:USERNAME):F"
icacls $authKeys /grant "SYSTEM:F"
```

**관리자 계정인 경우**, 대신 `C:\ProgramData\ssh\administrators_authorized_keys` (조직 전체 공용 파일 — 관리자 그룹 전원이 공유):
```powershell
$key = "<dev PC의 공개키 한 줄>"
$path = "C:\ProgramData\ssh\administrators_authorized_keys"
[System.IO.File]::WriteAllText($path, "$key`n", [System.Text.Encoding]::ASCII)
icacls $path /inheritance:r
icacls $path /grant "Administrators:F"
icacls $path /grant "SYSTEM:F"
```

두 경우 모두 **`Add-Content`/`notepad` 대신 `[System.IO.File]::WriteAllText`로 ASCII 인코딩으로 써야 한다** — PowerShell의 기본 텍스트 저장은 UTF-16(BOM 포함)이라 sshd가 파일을 파싱하지 못해 조용히 인증을 거부한다.

## 5. 그래도 안 되면: 원인을 직접 확인하는 법

권한/인코딩을 다 맞췄는데도 `Permission denied (publickey,password,keyboard-interactive)`가 뜨면, sshd를 포그라운드 디버그 모드로 띄워서 정확한 사유를 본다:

```powershell
# 방화벽에 임시 규칙 추가 (디버그용 포트)
New-NetFirewallRule -DisplayName "temp-ssh-debug-2222" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow

# 포그라운드로 실행 (연결 1회 받고 종료됨, 이 창은 그대로 둔다)
& "$env:WINDIR\System32\OpenSSH\sshd.exe" -d -p 2222
```

dev PC에서 `ssh -p 2222 <user>@<ip> "echo test"`로 접속을 시도하면, 이 창에 정확한 실패 사유(`user matched group list administrators`, `Could not open authorized keys '...'` 등)가 그대로 찍힌다. 확인 후 방화벽 규칙은 삭제:
```powershell
Remove-NetFirewallRule -DisplayName "temp-ssh-debug-2222"
```

---

여기까지 되면 dev PC에서 `ssh <username>@<ip주소>`로 접속 테스트해본다. 성공하면 다음 단계(코드 동기화 + 원격 실행 스크립트)로 넘어간다.

## 6. GPU가 최신(RTX 50 시리즈/Blackwell)인 경우

`run_remote.sh`는 `cu128` 인덱스로 torch를 설치한다. RTX 50 시리즈(sm_120)는 `cu121`/`cu124` 빌드에 커널이 없어 `torch.cuda.is_available()`은 `True`인데 실제 연산에서 `CUDA error: no kernel image is available`로 죽는다 — 증상이 헷갈리니 미리 알아두면 좋다. 더 오래된 GPU는 `cu121`이 더 가볍고 빠르니, `nvidia-smi`로 GPU 세대를 확인하고 필요시 `run_remote.sh`의 인덱스 URL을 낮춰도 된다.
