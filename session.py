"""Coding session simulation with tool calling.

Simulates the multi-turn tool-calling flow used by coding tools
like OpenCode and Cline (system prompt -> user request -> tool calls
-> tool results -> assistant response -> follow-up request -> ...).

Designed to generate large contexts (tens of thousands of tokens per
session) over 30+ turns to reproduce context contamination under
sustained async-scheduling load.
"""

import json
import random
import asyncio
import time
from openai import AsyncOpenAI
from user_profile import UserProfile


# ---------------------------------------------------------------------------
# Rough token estimator (avoids tiktoken dependency)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for mixed JP/KR/EN text."""
    return max(1, len(text) // 3)


def _estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across all messages."""
    total = 0
    for msg in messages:
        if msg.get("content"):
            total += _estimate_tokens(msg["content"])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("arguments"):
                    total += _estimate_tokens(tc["function"]["arguments"])
    return total


def _trim_messages_to_target(
    messages: list[dict],
    target_tokens: int = 120_000,
) -> list[dict]:
    """Keep system prompt + recent messages within target token budget.

    Always preserves:
    - messages[0] (system prompt)
    - messages[1] (initial user request)
    - The most recent messages

    Removes middle messages when total exceeds target.
    """
    if len(messages) <= 4:
        return messages

    total = _estimate_messages_tokens(messages)
    if total <= target_tokens:
        return messages

    # Keep system + initial user, then trim from oldest conversation messages
    preserved_start = messages[:2]
    rest = messages[2:]

    # Remove from the front of 'rest' until we fit
    while rest and _estimate_messages_tokens(preserved_start + rest) > target_tokens:
        # Remove in groups (assistant + tool results tend to come together)
        rest.pop(0)

    return preserved_start + rest


# ---------------------------------------------------------------------------
# Tool definitions (mimics coding tool capabilities)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for code patterns in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"}
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Project guidelines (large system prompt, realistic for coding tools)
# ---------------------------------------------------------------------------

def _build_project_guidelines(profile: UserProfile) -> str:
    """Generate a large project-context block for the system prompt.

    Content is fully in the user's language (Japanese or Korean) so that
    any cross-language leakage in the model's response is a reliable
    contamination signal — NOT caused by quoting the prompt itself.
    """
    if profile.language == "japanese":
        return f"""
## プロジェクト情報
- プロジェクトコード: {profile.project_code}
- 担当者: {profile.persona_name}
- 社内トラッキングID: {profile.canary_uuid}
- ビルドタグ: {profile.canary_build}
- メインクラス: {profile.unique_class}
- コンテキスト変数: {profile.unique_var}
- 定数: {profile.unique_const}

## コーディング規約
- すべてのパブリック関数にはdocstringを記載すること
- 変数名はsnake_caseを使用 (クラス名のみPascalCase)
- ログ出力には必ずプロジェクトコード {profile.project_code} をプレフィックスとして付与する
- エラーハンドリングは try/except で行い、例外は必ずログに記録する
- テストカバレッジは80%以上を維持すること
- コミットメッセージは "{profile.project_code}: <type>: <description>" 形式で記載
- セキュリティ上の理由から、社内ID ({profile.canary_uuid}) をログに直接出力しないこと
- 定数 {profile.unique_const} の変更は必ずコードレビューを経ること

## アーキテクチャ
```
{profile.project_code}/
├── src/
│   ├── main.py           メインエントリポイント ({profile.unique_class})
│   ├── utils.py           共通ユーティリティ
│   ├── config.py          設定管理 ({profile.unique_const})
│   ├── models.py          データモデル定義
│   ├── api.py             REST APIエンドポイント
│   ├── database.py        データベースアクセス層
│   ├── middleware.py      ミドルウェア (認証・ログ)
│   ├── cache.py           キャッシュ管理
│   ├── workers.py         バックグラウンドワーカー
│   └── validators.py      入力バリデーション
├── tests/
│   ├── test_main.py       {profile.unique_class} のユニットテスト
│   ├── test_utils.py      ユーティリティテスト
│   ├── test_api.py        APIエンドポイントテスト
│   ├── test_models.py     モデルテスト
│   ├── test_database.py   データベーステスト
│   └── conftest.py        テストフィクスチャ
├── docs/
│   ├── README.md          プロジェクト概要
│   ├── API.md             API仕様書
│   └── CHANGELOG.md       変更履歴
├── requirements.txt       依存パッケージ
├── Dockerfile             コンテナ定義
└── .github/workflows/     CI/CD設定
```

## 最近の変更履歴
- 2026-03-28: {profile.persona_name} が {profile.unique_class} のメモリリークを修正 (#{random.randint(100,999)})
- 2026-03-25: {profile.unique_const} のデフォルト値を変更 (#{random.randint(100,999)})
- 2026-03-20: database.py の接続プールを最適化 (#{random.randint(100,999)})
- 2026-03-15: middleware.py に認証トークン検証を追加 (#{random.randint(100,999)})
- 2026-03-10: cache.py にTTL管理機能を追加 (#{random.randint(100,999)})
- 2026-03-05: workers.py のリトライロジックを改善 (#{random.randint(100,999)})
- 2026-03-01: validators.py の入力チェックを強化 (#{random.randint(100,999)})

## 既知の問題
- {profile.unique_class} の初期化順序に問題あり (_setup_logging と _load_config)
- {profile.unique_const} の値が環境によって異なるケースが報告されている
- 高負荷時にdatabase.pyのコネクションプールが枯渇する場合がある
- cache.pyのTTL切れ時にレースコンディションが発生する可能性あり
"""
    elif profile.language == "korean":
        return f"""
## 프로젝트 정보
- 프로젝트 코드: {profile.project_code}
- 담당자: {profile.persona_name}
- 사내 트래킹 ID: {profile.canary_uuid}
- 빌드 태그: {profile.canary_build}
- 메인 클래스: {profile.unique_class}
- 컨텍스트 변수: {profile.unique_var}
- 상수: {profile.unique_const}

## 코딩 규약
- 모든 퍼블릭 함수에는 docstring을 작성할 것
- 변수명은 snake_case를 사용 (클래스명만 PascalCase)
- 로그 출력에는 반드시 프로젝트 코드 {profile.project_code} 를 접두사로 붙일 것
- 에러 핸들링은 try/except 로 수행하고, 예외는 반드시 로그에 기록할 것
- 테스트 커버리지는 80% 이상을 유지할 것
- 커밋 메시지는 "{profile.project_code}: <type>: <description>" 형식으로 작성
- 보안상의 이유로 사내 ID ({profile.canary_uuid}) 를 로그에 직접 출력하지 않을 것
- 상수 {profile.unique_const} 의 변경은 반드시 코드 리뷰를 거칠 것

## 아키텍처
```
{profile.project_code}/
├── src/
│   ├── main.py           메인 엔트리포인트 ({profile.unique_class})
│   ├── utils.py           공통 유틸리티
│   ├── config.py          설정 관리 ({profile.unique_const})
│   ├── models.py          데이터 모델 정의
│   ├── api.py             REST API 엔드포인트
│   ├── database.py        데이터베이스 접근 계층
│   ├── middleware.py      미들웨어 (인증 및 로그)
│   ├── cache.py           캐시 관리
│   ├── workers.py         백그라운드 워커
│   └── validators.py      입력 유효성 검사
├── tests/
│   ├── test_main.py       {profile.unique_class} 유닛 테스트
│   ├── test_utils.py      유틸리티 테스트
│   ├── test_api.py        API 엔드포인트 테스트
│   ├── test_models.py     모델 테스트
│   ├── test_database.py   데이터베이스 테스트
│   └── conftest.py        테스트 픽스처
├── docs/
│   ├── README.md          프로젝트 개요
│   ├── API.md             API 사양서
│   └── CHANGELOG.md       변경 이력
├── requirements.txt       의존 패키지
├── Dockerfile             컨테이너 정의
└── .github/workflows/     CI/CD 설정
```

## 최근 변경 이력
- 2026-03-28: {profile.persona_name} 이(가) {profile.unique_class} 의 메모리 누수를 수정 (#{random.randint(100,999)})
- 2026-03-25: {profile.unique_const} 의 기본값을 변경 (#{random.randint(100,999)})
- 2026-03-20: database.py 의 커넥션 풀을 최적화 (#{random.randint(100,999)})
- 2026-03-15: middleware.py 에 인증 토큰 검증을 추가 (#{random.randint(100,999)})
- 2026-03-10: cache.py 에 TTL 관리 기능을 추가 (#{random.randint(100,999)})
- 2026-03-05: workers.py 의 리트라이 로직을 개선 (#{random.randint(100,999)})
- 2026-03-01: validators.py 의 입력 체크를 강화 (#{random.randint(100,999)})

## 알려진 문제
- {profile.unique_class} 의 초기화 순서에 문제 있음 (_setup_logging 과 _load_config)
- {profile.unique_const} 의 값이 환경에 따라 다른 케이스가 보고됨
- 고부하 시 database.py 의 커넥션 풀이 고갈되는 경우가 있음
- cache.py 의 TTL 만료 시 레이스 컨디션이 발생할 가능성 있음
"""
    else:  # thai
        return f"""
## ข้อมูลโปรเจกต์
- รหัสโปรเจกต์: {profile.project_code}
- ผู้รับผิดชอบ: {profile.persona_name}
- ID ติดตามภายใน: {profile.canary_uuid}
- แท็กบิลด์: {profile.canary_build}
- คลาสหลัก: {profile.unique_class}
- ตัวแปรบริบท: {profile.unique_var}
- ค่าคงที่: {profile.unique_const}

## มาตรฐานการเขียนโค้ด
- ฟังก์ชันสาธารณะทุกตัวต้องมี docstring
- ชื่อตัวแปรใช้ snake_case (เฉพาะชื่อคลาสใช้ PascalCase)
- การแสดงผลบันทึกต้องมีรหัสโปรเจกต์ {profile.project_code} เป็นคำนำหน้าเสมอ
- การจัดการข้อผิดพลาดใช้ try/except และต้องบันทึกข้อยกเว้นลงบันทึก
- ความครอบคลุมของการทดสอบต้องรักษาไว้ที�� 80% ขึ้นไป
- ข้อความคอมมิตใช้รูปแบบ "{profile.project_code}: <type>: <description>"
- ด้วยเหตุผลด้านความปลอดภัย ID ภายใน ({profile.canary_uuid}) ต้องไม่แสดงในบันทึกโดยตรง
- การเปลี่ยนแปลงค่าคงที่ {profile.unique_const} ต้องผ่านการตรวจสอบโค้ด

## สถาปัตยกรรม
```
{profile.project_code}/
├── src/
│   ├── main.py           จุดเริ่มต้นหลัก ({profile.unique_class})
│   ├── utils.py           ยูทิลิตี้ทั่วไป
│   ├── config.py          การจัดการการกำหนดค่า ({profile.unique_const})
│   ├── models.py          คำจำกัดความของโมเดลข้อมูล
│   ├── api.py             จุดปลาย REST API
│   ├── database.py        ชั้นเข้าถึงฐานข้อมูล
│   ├── middleware.py      มิดเดิลแวร์ (การยืนยันตัวตนและบันทึก)
│   ├── cache.py           การจัดการแคช
│   ├── workers.py         เวิร์กเกอร์พื้นหลัง
│   └── validators.py      การตรวจสอบอินพุต
├── tests/
│   ├── test_main.py       การทดสอบหน่วย {profile.unique_class}
│   ├── test_utils.py      การทดสอบยูทิลิตี้
│   ├── test_api.py        การทดสอบจุดปลาย API
│   ├── test_models.py     การทดสอบโมเดล
│   ├── test_database.py   การทดสอบฐานข้อมูล
│   └── conftest.py        ฟิกซ์เจอร์ทดสอบ
├── docs/
│   ├── README.md          ภาพรวมโปรเจกต์
│   ├── API.md             ข้อกำหนด API
│   └── CHANGELOG.md       บันทึกการเปลี่ยนแปลง
├── requirements.txt       แพ็กเกจที่ต้องพึ่งพา
├── Dockerfile             คำจำกัดความคอนเทนเนอร์
└── .github/workflows/     การกำหนดค่า CI/CD
```

## บันทึกการเปลี่ยนแปลงล่าสุด
- 2026-03-28: {profile.persona_name} แก้ไขการรั่วไหลของหน่วยความจำใน {profile.unique_class} (#{random.randint(100,999)})
- 2026-03-25: เปลี่ยนค่าเริ่มต้นของ {profile.unique_const} (#{random.randint(100,999)})
- 2026-03-20: ปรับปรุงพูลการเชื่อมต่อใน database.py (#{random.randint(100,999)})
- 2026-03-15: เพิ่มการตรวจสอบโทเค็นยืนยันตัวตนใน middleware.py (#{random.randint(100,999)})
- 2026-03-10: เพิ่มฟังก์ชันจัดการ TTL ใน cache.py (#{random.randint(100,999)})
- 2026-03-05: ปรับปรุงตรรกะการลองใหม่ใน workers.py (#{random.randint(100,999)})
- 2026-03-01: เสริมความแข็งแกร่งการตรวจสอบอินพุตใน validators.py (#{random.randint(100,999)})

## ปัญหาที่ทราบ
- ลำดับการเริ่มต้นของ {profile.unique_class} มีปัญหา (_setup_logging และ _load_config)
- มีรายงานว่าค่าของ {profile.unique_const} แตกต่างกันในสภาพแวดล้อมต่างๆ
- พูลการเชื่อมต่อของ database.py อาจหมดในช่วงโหลดสูง
- อาจเกิดสภาวะแข่งขันเมื่อ TTL ของ cache.py หมดอายุ
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(profile: UserProfile) -> str:
    guidelines = _build_project_guidelines(profile)
    if profile.language == "japanese":
        return (
            f"あなたはプロジェクト {profile.project_code} の開発アシスタントです。\n"
            f"担当者: {profile.persona_name}\n"
            f"社内ID: {profile.canary_uuid}\n"
            f"ビルドタグ: {profile.canary_build}\n\n"
            f"あなたは利用可能なツールを使ってコードの調査・修正を行います。\n"
            f"必ず日本語で回答してください。回答にはプロジェクト名・担当者名・社内IDを含めてください。\n"
            f"ユーザーの指示に従って、ツールを積極的に使用してコードを調査してください。\n"
            f"{guidelines}"
        )
    elif profile.language == "korean":
        return (
            f"당신은 프로젝트 {profile.project_code} 의 개발 어시스턴트입니다.\n"
            f"담당자: {profile.persona_name}\n"
            f"사내ID: {profile.canary_uuid}\n"
            f"빌드태그: {profile.canary_build}\n\n"
            f"사용 가능한 도구를 사용하여 코드를 조사하고 수정합니다.\n"
            f"반드시 한국어로 답변하세요. 답변에 프로젝트명, 담당자명, 사내ID를 포함하세요.\n"
            f"사용자의 지시에 따라 도구를 적극적으로 사용하여 코드를 조사하세요.\n"
            f"{guidelines}"
        )
    else:  # thai
        return (
            f"คุณเป็นผู��ช่วยพัฒนาโปรเจกต์ {profile.project_code}\n"
            f"ผู้รับผิดชอบ: {profile.persona_name}\n"
            f"ID ภายใน: {profile.canary_uuid}\n"
            f"แท็กบิลด์: {profile.canary_build}\n\n"
            f"คุณใช้เครื่องมือที่มีเพื่อตรวจสอบและแก้ไขโค้ด\n"
            f"กรุณาตอบเป็นภาษาไทยเท่านั้น ในคำตอบให้ระบุชื่อโปรเจกต์ ชื่อผู้รับผิดชอบ และ ID ภายใน\n"
            f"กรุณาใช้เครื่องมือเพื่อตรวจสอบโค้ดตามคำแนะนำของผู้ใช้\n"
            f"{guidelines}"
        )


def build_user_message(profile: UserProfile) -> str:
    if profile.language == "japanese":
        return (
            f"プロジェクト {profile.project_code} の包括的なコードレビューと"
            f"バグ修正を行ってください。\n\n"
            f"まず src/main.py を読み込んで、クラス {profile.unique_class} の"
            f"初期化処理のバグを調査してください。その後、関連するファイルも"
            f"順次確認していきます。\n\n"
            f"ファイルを読み込む際は read_file ツールを使ってください。"
            f"コード検索には search_code ツールを使ってください。"
        )
    elif profile.language == "korean":
        return (
            f"프로젝트 {profile.project_code} 의 포괄적인 코드 리뷰와 "
            f"버그 수정을 진행해주세요.\n\n"
            f"먼저 src/main.py 를 읽어서 클래스 {profile.unique_class} 의 "
            f"초기화 처리 버그를 조사해주세요. 그 후 관련 파일도 "
            f"순차적으로 확인하겠습니다.\n\n"
            f"파일을 읽을 때는 read_file 도구를 사용해주세요. "
            f"코드 검색에는 search_code 도구를 사용해주세요."
        )
    else:  # thai
        return (
            f"กรุณาทำการตรวจสอบโค้ดอย่างครอบคลุมและแก้ไขบั๊กสำหรับ"
            f"โปรเจกต์ {profile.project_code}\n\n"
            f"ก่อนอื่นกรุณาอ่าน src/main.py และตรวจสอบบั๊กในการเริ่มต้น"
            f"คลาส {profile.unique_class} จากนั้นเราจะตรวจสอบไฟล์ที่เกี่ยวข้อง"
            f"ตามลำดับ\n\n"
            f"เมื่ออ่านไฟล์กรุณาใช้เครื่องมือ read_file "
            f"สำหรับค้นหาโค้ดกรุณาใช้เครื่องมือ search_code"
        )


# ---------------------------------------------------------------------------
# Follow-up messages (injected when model stops calling tools)
# ---------------------------------------------------------------------------

def _build_follow_up_messages(profile: UserProfile) -> list[str]:
    """Return 15+ follow-up user messages to keep the session going for 30+ turns.

    Each follow-up asks the model to investigate a different file or perform
    a different task, encouraging tool usage and growing the context.
    """
    if profile.language == "japanese":
        return [
            (
                f"次に tests/test_main.py を読んで、{profile.unique_class} のテストケースを"
                f"確認してください。テストが十分にカバーされているか見てください。"
            ),
            (
                f"src/utils.py も読んで、{profile.project_code} の共通ユーティリティに"
                f"問題がないか確認してください。"
            ),
            (
                f"src/config.py の設定値を確認してください。特に {profile.unique_const} の"
                f"値が正しいか、環境ごとの差異がないか見てください。"
            ),
            (
                f"エラーハンドリングを改善してください。まず 'except' をキーワードに"
                f"コード検索して、現在のエラーハンドリングの状況を把握してください。"
            ),
            (
                f"テストを実行して結果を確認してください。"
                f"コマンド 'python -m pytest tests/ -v' を実行してください。"
            ),
            (
                f"src/models.py を読んで、データモデルの定義を確認してください。"
                f"{profile.unique_class} が使うモデルに問題がないか見てください。"
            ),
            (
                f"ログ出力を改善してください。まず 'logging' をキーワードに"
                f"コード検索して、現在のログ設定を確認してください。"
            ),
            (
                f"src/api.py を読んで、APIエンドポイントの実装を確認してください。"
                f"バリデーション漏れやセキュリティの問題がないか見てください。"
            ),
            (
                f"src/database.py を読んで、データベースアクセスのコードを確認してください。"
                f"コネクションプールの管理に問題がないか見てください。"
            ),
            (
                f"src/middleware.py を読んで、ミドルウェアの処理を確認してください。"
                f"認証トークンの検証ロジックに問題がないか確認してください。"
            ),
            (
                f"src/cache.py を読んで、キャッシュ管理の実装を確認してください。"
                f"TTL管理やレースコンディションの問題がないか見てください。"
            ),
            (
                f"src/workers.py を読んで、バックグラウンドワーカーの実装を確認してください。"
                f"リトライロジックやエラーハンドリングに問題がないか見てください。"
            ),
            (
                f"src/validators.py を読んで、入力バリデーションの実装を確認してください。"
                f"セキュリティ上の問題がないか確認してください。"
            ),
            (
                f"ファイル一覧を確認して、まだ見ていないファイルがないかチェックしてください。"
                f"その後、tests/conftest.py を読んでテストフィクスチャも確認してください。"
            ),
            (
                f"これまでの調査結果をまとめて、修正が必要な箇所のリストを作成してください。"
                f"プロジェクト {profile.project_code}、担当者 {profile.persona_name}、"
                f"社内ID {profile.canary_uuid} を含めて報告してください。"
            ),
        ]
    elif profile.language == "korean":
        return [
            (
                f"다음으로 tests/test_main.py 를 읽어서 {profile.unique_class} 의 테스트 케이스를 "
                f"확인해주세요. 테스트가 충분히 커버되고 있는지 확인해주세요."
            ),
            (
                f"src/utils.py 도 읽어서 {profile.project_code} 의 공통 유틸리티에 "
                f"문제가 없는지 확인해주세요."
            ),
            (
                f"src/config.py 의 설정값을 확인해주세요. 특히 {profile.unique_const} 의 "
                f"값이 올바른지, 환경별 차이가 없는지 확인해주세요."
            ),
            (
                f"에러 핸들링을 개선해주세요. 먼저 'except' 를 키워드로 "
                f"코드 검색을 해서 현재 에러 핸들링 상황을 파악해주세요."
            ),
            (
                f"테스트를 실행하고 결과를 확인해주세요. "
                f"'python -m pytest tests/ -v' 명령을 실행해주세요."
            ),
            (
                f"src/models.py 를 읽어서 데이터 모델 정의를 확인해주세요. "
                f"{profile.unique_class} 가 사용하는 모델에 문제가 없는지 확인해주세요."
            ),
            (
                f"로그 출력을 개선해주세요. 먼저 'logging' 을 키워드로 "
                f"코드 검색을 해서 현재 로그 설정을 확인해주세요."
            ),
            (
                f"src/api.py 를 읽어서 API 엔드포인트 구현을 확인해주세요. "
                f"유효성 검사 누락이나 보안 문제가 없는지 확인해주세요."
            ),
            (
                f"src/database.py 를 읽어서 데이터베이스 접근 코드를 확인해주세요. "
                f"커넥션 풀 관리에 문제가 없는지 확인해주세요."
            ),
            (
                f"src/middleware.py 를 읽어서 미들웨어 처리를 확인해주세요. "
                f"인증 토큰 검증 로직에 문제가 없는지 확인해주세요."
            ),
            (
                f"src/cache.py 를 읽어서 캐시 관리 구현을 확인해주세요. "
                f"TTL 관리나 레이스 컨디션 문제가 없는지 확인해주세요."
            ),
            (
                f"src/workers.py 를 읽어서 백그라운드 워커 구현을 확인해주세요. "
                f"리트라이 로직이나 에러 핸들링에 문제가 없는지 확인해주세요."
            ),
            (
                f"src/validators.py 를 읽어서 입력 유효성 검사 구현을 확인해주세요. "
                f"보안상의 문제가 없는지 확인해주세요."
            ),
            (
                f"파일 목록을 확인하고 아직 확인하지 않은 파일이 없는지 체크해주세요. "
                f"그 후 tests/conftest.py 를 읽어서 테스트 픽스처도 확인해주세요."
            ),
            (
                f"지금까지의 조사 결과를 정리하고 수정이 필요한 부분의 목록을 작성해주세요. "
                f"프로젝트 {profile.project_code}, 담당자 {profile.persona_name}, "
                f"사내ID {profile.canary_uuid} 를 포함해서 보고해주세요."
            ),
        ]
    else:  # thai
        return [
            (
                f"ต่อไปกรุณาอ่าน tests/test_main.py และตรวจสอบกรณีทดสอบของ {profile.unique_class} "
                f"กรุณาตรวจสอบว่าการทดสอบครอบคลุมเพียงพอหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/utils.py ด้วย และตรวจสอบว่ายูทิลิตี้ทั่วไปของ "
                f"{profile.project_code} มีปัญหาหรือไม่"
            ),
            (
                f"กรุณาตรวจสอบค่าการกำหนดค่าใน src/config.py โดยเฉพาะ {profile.unique_const} "
                f"ว่าค่าถูกต้องหรือไม่ และมีความแตกต่างระหว่างสภาพแวดล้อมหรือไม่"
            ),
            (
                f"กรุณาปรับปรุงการจัดการข้อผิดพลาด ก่อนอื่นค้นหาโค้ดด้วยคำสำคัญ 'except' "
                f"เพื่อทำความเข้าใจสถานการณ์การจัดการข้อผิดพลาดปัจจุบัน"
            ),
            (
                f"กรุณาเรียกใช้การทดสอบและตรวจสอบผลลัพธ์ "
                f"กรุณาเรียกใช้คำสั่ง 'python -m pytest tests/ -v'"
            ),
            (
                f"กรุณาอ่าน src/models.py และตรวจสอบคำจำกัดความของโมเดลข้อมูล "
                f"กรุณาตรวจสอบว่าโมเดลที่ {profile.unique_class} ใช้มีปัญหาหรือไม่"
            ),
            (
                f"กรุณาปรับปรุงการแสดงผลบันทึก ก่อนอื่นค้นหาโค้ดด้วยคำสำคัญ 'logging' "
                f"เพื่อตรวจสอบการกำหนดค่าบันทึกปัจจุบัน"
            ),
            (
                f"กรุณาอ่าน src/api.py และตรวจสอบการดำเนินการจุดปลาย API "
                f"กรุณาตรวจสอบว่ามีการตรวจสอบที่ขาดหายไปหรือปัญหาด้านความปลอดภัยหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/database.py และตรวจสอบโค้ดเข้าถึงฐานข้อมูล "
                f"กรุณาตรวจสอบว่าการจัดการพูลการเชื่อมต่อมีปัญหาหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/middleware.py และตรวจสอบการประมวลผลมิดเดิลแวร์ "
                f"กรุณาตรวจสอบว่าตรรกะการตรวจสอบโทเค็นยืนยันตัวตนมีปัญหาหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/cache.py และตรวจสอบการดำเนินการจัดการแคช "
                f"กรุณาตรวจสอบว่ามีปัญหาการจัดการ TTL หรือสภาวะแข่งขันหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/workers.py และตรวจสอบการดำเนินการเวิร์กเกอร์พื้นหลัง "
                f"กรุณาตรวจสอบว่ามีปัญหาตรรกะการลองใหม่หรือการจัดการข้อผิดพลาดหรือไม่"
            ),
            (
                f"กรุณาอ่าน src/validators.py และตรวจสอบการดำเนินการตรวจสอบอินพุต "
                f"กรุณาตรวจสอบว่ามีปัญหาด้านความปลอดภัยหรือไม่"
            ),
            (
                f"กรุณาตรวจสอบรายการไฟล์และตรวจว่ายังมีไฟล์ที่ยังไม่ได้ตรวจสอบหรือไม่ "
                f"จากนั้นกรุณาอ่าน tests/conftest.py และตรวจสอบฟิกซ์เจอร์ทดสอบด้วย"
            ),
            (
                f"กรุณาสรุปผลการตรวจสอบจนถึงตอนนี้ และสร้างรายการส่วนที่ต้องแก้ไข "
                f"กรุณารวมโปรเจกต์ {profile.project_code} ผู้รับผิดชอบ {profile.persona_name} "
                f"ID ภายใน {profile.canary_uuid} ในรายงาน"
            ),
        ]


# ---------------------------------------------------------------------------
# Fake tool result generation
# ---------------------------------------------------------------------------

def _generate_padding_code(profile: UserProfile, lines: int = 150) -> str:
    """Generate fake code blocks with embedded canary tokens.

    Default 150 lines produces ~12,000 chars (~4,000 tokens) per file read.
    With 10+ reads over 30 turns, total context reaches ~60,000-120,000 tokens.
    """
    code_lines = []
    methods = [
        "initialize", "process", "validate", "transform", "execute",
        "compute", "aggregate", "dispatch", "serialize", "optimize",
        "configure", "cleanup", "migrate", "rollback", "checkpoint",
    ]
    for i in range(lines):
        method = methods[i % len(methods)]
        if i % 20 == 0:
            code_lines.append(f"    # {profile.project_code} - {profile.persona_name}")
            code_lines.append(f"    # Ref: {profile.canary_uuid}")
            code_lines.append(f"    # Build: {profile.canary_build}")
        if i % 10 == 0:
            code_lines.append("")
            code_lines.append(f"    def {method}_{i}(self, {profile.unique_var}=None):")
            code_lines.append(f'        """{profile.project_code} - step {i}')
            code_lines.append(f"        Maintained by {profile.persona_name}")
            code_lines.append(f'        Ref: {profile.canary_uuid}"""')
            code_lines.append(
                f"        {profile.unique_var} = {profile.unique_var} or self._{profile.unique_var}"
            )
            code_lines.append(
                f"        self.logger.debug('{profile.project_code}:{method}_{i}:start')"
            )
        else:
            code_lines.append(
                f"        result_{i} = self._{method}("
            )
            code_lines.append(
                f"            step={i},"
            )
            code_lines.append(
                f"            tag='{profile.canary_build}',"
            )
            code_lines.append(
                f"            project='{profile.project_code}',"
            )
            code_lines.append(
                f"        )")
            const_val = profile.unique_const.split("_")[-1]
            code_lines.append(f"        if result_{i} > {const_val}:")
            code_lines.append(
                f"            self.logger.warning("
            )
            code_lines.append(
                f"                '{profile.project_code}: threshold at step {i}, "
                f"ref={profile.canary_uuid}'"
            )
            code_lines.append(
                f"            )"
            )
            code_lines.append(
                f"            self.{profile.unique_var}['alert_{i}'] = True"
            )
    return "\n".join(code_lines)


def _generate_search_snippets(profile: UserProfile, query: str, count: int = 8) -> str:
    """Generate large search result with code context snippets."""
    files = [
        ("src/main.py", "class definition", [
            f"class {profile.unique_class}:",
            f'    """Core engine for {profile.project_code}"""',
            f'    MAINTAINER = "{profile.persona_name}"',
            f'    INTERNAL_ID = "{profile.canary_uuid}"',
        ]),
        ("src/utils.py", "utility function", [
            f"def get_project_config(project_code='{profile.project_code}'):",
            f'    """Load config for {profile.project_code}"""',
            f"    {profile.unique_var} = load_defaults()",
            f"    {profile.unique_var}['build'] = '{profile.canary_build}'",
        ]),
        ("src/config.py", "configuration", [
            f"{profile.unique_const} = {random.randint(1000, 9999)}",
            f"PROJECT_CODE = '{profile.project_code}'",
            f"MAINTAINER = '{profile.persona_name}'",
            f"BUILD_TAG = '{profile.canary_build}'",
        ]),
        ("src/database.py", "database layer", [
            f"class {profile.unique_class}DB:",
            f'    """Database adapter for {profile.project_code}"""',
            f"    def __init__(self, {profile.unique_var}=None):",
            f"        self.conn_id = '{profile.canary_uuid}'",
        ]),
        ("src/api.py", "API endpoint", [
            f"@app.route('/api/{profile.project_code.lower()}')",
            f"def handle_{profile.unique_var}(request):",
            f'    """API handler - {profile.persona_name}"""',
            f"    tracking_id = '{profile.canary_build}'",
        ]),
        ("src/middleware.py", "middleware", [
            f"class {profile.unique_class}Auth:",
            f"    PROJECT = '{profile.project_code}'",
            f"    def verify(self, token, ref='{profile.canary_uuid}'):",
            f"        return self._check(token, '{profile.canary_build}')",
        ]),
        ("src/cache.py", "cache management", [
            f"class {profile.unique_class}Cache:",
            f"    TTL = {profile.unique_const.split('_')[-1]}",
            f"    PROJECT = '{profile.project_code}'",
            f"    def get(self, key, {profile.unique_var}=None):",
        ]),
        ("src/workers.py", "background worker", [
            f"class {profile.unique_class}Worker:",
            f'    """Worker for {profile.project_code} - {profile.persona_name}"""',
            f"    BUILD = '{profile.canary_build}'",
            f"    REF = '{profile.canary_uuid}'",
        ]),
    ]

    lines = [f"Found {min(count, len(files)) * 4} matches for '{query}':\n"]
    for filepath, desc, snippets in files[:count]:
        line_num = random.randint(1, 200)
        lines.append(f"── {filepath} ({desc}) ──")
        for j, snippet in enumerate(snippets):
            lines.append(f"  {line_num + j}│ {snippet}")
        lines.append("")
    return "\n".join(lines)


def _generate_test_output(profile: UserProfile) -> str:
    """Generate large test execution output."""
    test_files = [
        "test_main.py", "test_utils.py", "test_api.py",
        "test_models.py", "test_database.py",
    ]
    lines = [
        f"$ python -m pytest tests/ -v",
        f"========================= test session starts =========================",
        f"platform linux -- Python 3.11.5, pytest-7.4.0",
        f"project: {profile.project_code}",
        f"maintainer: {profile.persona_name}",
        f"build: {profile.canary_build}",
        f"ref: {profile.canary_uuid}",
        f"collected 156 items",
        f"",
    ]
    for tf in test_files:
        for i in range(25):
            status = "PASSED" if random.random() > 0.08 else "FAILED"
            method = random.choice([
                "test_init", "test_process", "test_validate",
                "test_config", "test_error", "test_edge_case",
                "test_performance", "test_integration",
            ])
            lines.append(
                f"tests/{tf}::{method}_{i}[{profile.project_code}] {status}"
            )
            if status == "FAILED":
                lines.append(f"    AssertionError: {profile.unique_class}.{method} "
                             f"returned unexpected value")
                lines.append(f"    ref: {profile.canary_uuid}")
                lines.append(f"    {profile.unique_const} = "
                             f"{random.randint(1000, 9999)}")
    total = len(test_files) * 25
    failed = sum(1 for l in lines if "FAILED" in l) // 3
    lines.extend([
        f"",
        f"========================= results =========================",
        f"{total - failed} passed, {failed} failed -- {profile.project_code}",
        f"maintainer: {profile.persona_name}",
        f"build: {profile.canary_build}",
    ])
    return "\n".join(lines)


def generate_tool_result(tool_call: dict, profile: UserProfile) -> str:
    """Generate a fake tool result with canary tokens embedded."""
    name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except (json.JSONDecodeError, TypeError):
        args = {}

    if name == "read_file":
        path = args.get("path", "src/main.py")
        padding = _generate_padding_code(profile, lines=150)
        return (
            f"# {profile.project_code} / {path}\n"
            f"# Author: {profile.persona_name}\n"
            f"# Internal-ID: {profile.canary_uuid}\n"
            f"# Build-Tag: {profile.canary_build}\n"
            f"# Class: {profile.unique_class}\n"
            f"# Var: {profile.unique_var}\n"
            f"# Const: {profile.unique_const}\n\n"
            f"import logging\n"
            f"import os\n"
            f"import json\n"
            f"from typing import Optional, Dict, Any, List\n"
            f"from dataclasses import dataclass, field\n\n"
            f'logger = logging.getLogger("{profile.project_code}")\n\n'
            f"{profile.unique_const} = {random.randint(1000, 9999)}\n"
            f"_PROJECT = '{profile.project_code}'\n"
            f"_MAINTAINER = '{profile.persona_name}'\n"
            f"_BUILD = '{profile.canary_build}'\n\n\n"
            f"@dataclass\n"
            f"class {profile.unique_class}Config:\n"
            f'    """Configuration for {profile.unique_class} ({profile.project_code})"""\n'
            f"    project: str = '{profile.project_code}'\n"
            f"    maintainer: str = '{profile.persona_name}'\n"
            f"    internal_id: str = '{profile.canary_uuid}'\n"
            f"    build_tag: str = '{profile.canary_build}'\n"
            f"    max_value: int = {random.randint(1000, 9999)}\n\n\n"
            f"class {profile.unique_class}:\n"
            f'    """Core engine for {profile.project_code}\n'
            f"    Maintained by {profile.persona_name}\n"
            f"    Internal-ID: {profile.canary_uuid}\n"
            f"    Build-Tag: {profile.canary_build}\n"
            f'    """\n\n'
            f'    PROJECT = "{profile.project_code}"\n'
            f'    MAINTAINER = "{profile.persona_name}"\n'
            f'    INTERNAL_ID = "{profile.canary_uuid}"\n'
            f"    {profile.unique_const} = {random.randint(1000, 9999)}\n\n"
            f"    def __init__(self, config: Optional[{profile.unique_class}Config] = None):\n"
            f"        self.{profile.unique_var}: Dict[str, Any] = {{}}\n"
            f'        self.build_tag = "{profile.canary_build}"\n'
            f"        self._initialized = False\n"
            f"        self.config = config or {profile.unique_class}Config()\n"
            f"        # BUG: initialization order is wrong\n"
            f"        self._setup_logging()\n"
            f"        self._load_config()  # should be called before _setup_logging\n\n"
            f"    def _setup_logging(self):\n"
            f"        self.logger = logging.getLogger(\n"
            f'            f"{profile.project_code}.{{self.__class__.__name__}}"\n'
            f"        )\n\n"
            f"    def _load_config(self):\n"
            f'        self.{profile.unique_var}["project"] = "{profile.project_code}"\n'
            f'        self.{profile.unique_var}["maintainer"] = "{profile.persona_name}"\n'
            f'        self.{profile.unique_var}["internal_id"] = "{profile.canary_uuid}"\n'
            f'        self.{profile.unique_var}["build"] = "{profile.canary_build}"\n'
            f"        self._initialized = True\n\n"
            f"{padding}\n"
        )

    elif name == "search_code":
        query = args.get("query", "bug")
        return _generate_search_snippets(profile, query)

    elif name == "write_file":
        content = args.get("content", "")
        path = args.get("path", "unknown")
        return f"Successfully wrote {len(content)} bytes to {path}"

    elif name == "list_files":
        return (
            f"{profile.project_code}/\n"
            f"├── src/\n"
            f"│   ├── main.py           (8.5 KB) {profile.unique_class} - main entry\n"
            f"│   ├── utils.py          (3.2 KB) {profile.project_code} utilities\n"
            f"│   ├── config.py         (2.1 KB) Configuration - {profile.unique_const}\n"
            f"│   ├── models.py         (4.7 KB) Data models\n"
            f"│   ├── api.py            (6.3 KB) REST API endpoints\n"
            f"│   ├── database.py       (5.8 KB) Database access layer\n"
            f"│   ├── middleware.py      (3.4 KB) Auth & logging middleware\n"
            f"│   ├── cache.py          (2.9 KB) Cache management\n"
            f"│   ├── workers.py        (4.1 KB) Background workers\n"
            f"│   └── validators.py     (2.6 KB) Input validation\n"
            f"├── tests/\n"
            f"│   ├── test_main.py      (5.2 KB) {profile.unique_class} tests\n"
            f"│   ├── test_utils.py     (2.8 KB)\n"
            f"│   ├── test_api.py       (4.5 KB)\n"
            f"│   ├── test_models.py    (3.1 KB)\n"
            f"│   ├── test_database.py  (3.9 KB)\n"
            f"│   └── conftest.py       (1.8 KB) Fixtures\n"
            f"├── docs/\n"
            f"│   ├── README.md         Project: {profile.project_code}\n"
            f"│   ├── API.md            API specification\n"
            f"│   └── CHANGELOG.md      Maintainer: {profile.persona_name}\n"
            f"├── requirements.txt\n"
            f"├── Dockerfile\n"
            f"└── .github/workflows/\n"
            f"    └── ci.yml            Build: {profile.canary_build}\n"
        )

    elif name == "run_command":
        cmd = args.get("command", "echo ok")
        if "pytest" in cmd or "test" in cmd:
            return _generate_test_output(profile)
        return (
            f"$ {cmd}\n"
            f"Running for {profile.project_code}...\n"
            f"{profile.unique_class}: operation completed\n"
            f"Maintainer: {profile.persona_name}\n"
            f"Build tag: {profile.canary_build}\n"
            f"Ref: {profile.canary_uuid}\n"
        )

    return "OK"


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------

def _extract_reasoning(msg) -> str | None:
    """Extract reasoning/thinking content from various possible locations.

    Kimi K2.5 returns thinking in the 'reasoning' field (not 'reasoning_content').
    We check both for compatibility.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if val:
            return val
    if hasattr(msg, "model_extra") and msg.model_extra:
        for key in ("reasoning_content", "reasoning"):
            val = msg.model_extra.get(key)
            if val:
                return val
    return None


def _normalize_response(response) -> dict:
    """Normalize an OpenAI API response to a standard dict format."""
    choice = response.choices[0]
    msg = choice.message
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return {
        "content": msg.content,
        "reasoning": _extract_reasoning(msg),
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason,
    }


def _extract_streaming_reasoning(delta) -> str | None:
    """Extract reasoning from a streaming delta.

    Checks both 'reasoning_content' and 'reasoning' for Kimi K2.5 compatibility.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(delta, attr, None)
        if val:
            return val
    if hasattr(delta, "model_extra") and delta.model_extra:
        for key in ("reasoning_content", "reasoning"):
            val = delta.model_extra.get(key)
            if val:
                return val
    return None


async def _streaming_request(client: AsyncOpenAI, **kwargs) -> dict:
    """Handle a streaming API request and collect the full response."""
    kwargs["stream"] = True
    stream = await client.chat.completions.create(**kwargs)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    finish_reason = None

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta and delta.content:
            content_parts.append(delta.content)

        if delta:
            r = _extract_streaming_reasoning(delta)
            if r:
                reasoning_parts.append(r)

        if delta and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )

    tool_calls = None
    if tool_calls_acc:
        tool_calls = [tool_calls_acc[idx] for idx in sorted(tool_calls_acc)]

    return {
        "content": "".join(content_parts) if content_parts else None,
        "reasoning": "".join(reasoning_parts) if reasoning_parts else None,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


# ---------------------------------------------------------------------------
# Session simulation
# ---------------------------------------------------------------------------

async def simulate_session(
    profile: UserProfile,
    base_url: str,
    model: str,
    max_turns: int = 40,
    streaming: bool = False,
    timeout: float = 300.0,
    extra_body: dict | None = None,
    on_turn_complete=None,
    context_target_tokens: int = 120_000,
    stop_event: asyncio.Event | None = None,
) -> dict:
    """Simulate a full coding-tool session for one user.

    Follows the real flow:
      system prompt (large) -> user request -> model calls tools ->
      client sends fake results (large) -> model responds ->
      follow-up user request -> model calls more tools -> ...

    When the model stops calling tools, a follow-up user message is
    injected to keep the session going.  This continues until max_turns
    API calls are made or all follow-up messages are exhausted.

    A typical session produces 30-40 API calls, exchanging tens of
    thousands of tokens in accumulated context.
    """
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=timeout,
    )

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(profile)},
        {"role": "user", "content": build_user_message(profile)},
    ]

    follow_ups = _build_follow_up_messages(profile)
    follow_up_idx = 0
    turns: list[dict] = []

    try:
        for turn_idx in range(max_turns):
            # Check stop_event between turns for fast shutdown
            if stop_event and stop_event.is_set():
                break

            default_extra = {
                "chat_template_kwargs": {"enable_thinking": True},
            }
            if extra_body:
                default_extra.update(extra_body)
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "extra_body": default_extra,
            }

            # Measure context size sent in this request
            context_chars = sum(
                len(m.get("content") or "") for m in messages
            )
            context_tokens_est = _estimate_messages_tokens(messages)

            request_start = time.monotonic()
            try:
                if streaming:
                    resp = await _streaming_request(client, **kwargs)
                else:
                    raw = await client.chat.completions.create(**kwargs)
                    resp = _normalize_response(raw)
            except asyncio.CancelledError:
                break
            request_duration = time.monotonic() - request_start

            # Re-check stop_event after potentially long API call
            if stop_event and stop_event.is_set():
                break

            # Response size (content + reasoning)
            response_chars = (
                len(resp.get("content") or "")
                + len(resp.get("reasoning") or "")
            )

            turn_data = {
                "turn": turn_idx,
                "content": resp["content"],
                "reasoning": resp["reasoning"],
                "tool_calls": resp["tool_calls"],
                "finish_reason": resp["finish_reason"],
                "context_chars": context_chars,
                "context_tokens_est": context_tokens_est,
                "message_count": len(messages),
                "request_duration_s": round(request_duration, 2),
                "response_chars": response_chars,
            }
            turns.append(turn_data)

            # Real-time detection callback
            if on_turn_complete:
                await on_turn_complete(profile, turn_data)

            # Build assistant message for conversation history
            asst_msg: dict = {"role": "assistant", "content": resp["content"]}
            if resp["tool_calls"]:
                asst_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in resp["tool_calls"]
                ]
            messages.append(asst_msg)

            if resp["tool_calls"]:
                # Send fake tool results back (contamination tends to
                # appear in the response *after* tool results)
                for tc in resp["tool_calls"]:
                    result = generate_tool_result(tc, profile)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                # Sliding window: trim old messages to keep context
                # around 100K-150K tokens, preventing unbounded growth
                messages = _trim_messages_to_target(
                    messages, target_tokens=context_target_tokens,
                )
            else:
                # Model gave a final response — inject follow-up to
                # keep the session going (mimics user sending next task)
                if follow_up_idx < len(follow_ups):
                    messages.append({
                        "role": "user",
                        "content": follow_ups[follow_up_idx],
                    })
                    follow_up_idx += 1
                else:
                    break  # All follow-ups exhausted

        return {
            "user_id": profile.user_id,
            "language": profile.language,
            "persona": profile.persona_name,
            "project": profile.project_code,
            "turns": turns,
            "total_turns": len(turns),
            "follow_ups_used": follow_up_idx,
            "error": None,
        }

    except Exception as e:
        return {
            "user_id": profile.user_id,
            "language": profile.language,
            "persona": profile.persona_name,
            "project": profile.project_code,
            "turns": turns,
            "total_turns": len(turns),
            "follow_ups_used": follow_up_idx,
            "error": str(e),
        }
