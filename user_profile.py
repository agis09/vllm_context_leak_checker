"""User profile generation with canary tokens for contamination testing."""

import uuid
import random
from dataclasses import dataclass


JAPANESE_PERSONAS = [
    "高橋雄一", "渡辺美咲", "伊藤健太", "中村さくら", "小林洋介",
    "田中真由美", "山本大輔", "加藤恵子", "吉田翔太", "佐々木あかり",
    "松本隆志", "井上優子", "木村和也", "林なつみ", "清水拓海",
    "山口千晴", "阿部慶介", "森下由美", "池田光", "橋本香織",
]

KOREAN_PERSONAS = [
    "김민준", "이서연", "박지훈", "최수빈", "정도윤",
    "강하은", "조현우", "윤지아", "임태현", "한소율",
    "신우진", "권예린", "오승민", "배나은", "류건호",
    "송다인", "홍재원", "문서진", "양시우", "전하린",
]

THAI_PERSONAS = [
    "สมชาย วงศ์สกุล", "สุภาพร แสงทอง", "ธนกร เจริญสุข", "พิมพ์ลดา รุ่งเรือง", "วิชัย สุขสมบูรณ์",
    "นภาพร ศรีสว่าง", "อนุชา พงศ์ไพบูลย์", "กัญญา ประเสริฐ", "ปิยะ มั่นคง", "ดวงใจ ชัยวัฒน์",
    "สุรชัย อมรรัตน์", "วรรณา สิริมงคล", "เกียรติ บุญเลิศ", "มาลี ดำรงค์", "ชัยวัฒน์ ปัญญา",
    "จันทร์เพ็ญ วิเศษ", "ประสิทธิ์ กิจเจริญ", "สายฝน อุดมทรัพย์", "นิรันดร์ สุวรรณ", "รัตนา พูลสวัสดิ์",
]

LANGUAGES = ["japanese", "korean", "thai"]

PROJECT_PREFIXES = [
    "FALCON", "EAGLE", "HAWK", "PHOENIX", "CONDOR",
    "DRAGON", "TIGER", "WOLF", "BEAR", "LION",
    "VIPER", "COBRA", "RAVEN", "CRANE", "STORM",
    "BLAZE", "FROST", "THUNDER", "SHADOW", "CRYSTAL",
]

CLASS_PREFIXES = [
    "Quantum", "Nebula", "Prism", "Helix", "Vortex",
    "Zenith", "Apex", "Nova", "Pulse", "Nexus",
    "Cipher", "Vector", "Matrix", "Flux", "Omega",
    "Delta", "Sigma", "Theta", "Lambda", "Kappa",
]

PERSONA_POOLS = {
    "japanese": JAPANESE_PERSONAS,
    "korean": KOREAN_PERSONAS,
    "thai": THAI_PERSONAS,
}


@dataclass
class UserProfile:
    user_id: str
    language: str  # "japanese", "korean", or "thai"
    persona_name: str
    project_code: str
    canary_uuid: str
    canary_build: str
    unique_class: str
    unique_var: str
    unique_const: str

    @property
    def all_canary_tokens(self) -> dict:
        """All canary tokens that can be used for contamination detection."""
        return {
            "canary_uuid": self.canary_uuid,
            "canary_build": self.canary_build,
            "persona_name": self.persona_name,
            "project_code": self.project_code,
            "unique_class": self.unique_class,
            "unique_var": self.unique_var,
            "unique_const": self.unique_const,
        }


def generate_profiles(users_per_lang: int, seed: int | None = None) -> list[UserProfile]:
    """Generate unique user profiles with canary tokens.

    Creates *users_per_lang* users for each of the 3 languages
    (japanese, korean, thai), for a total of users_per_lang * 3 profiles.
    """
    if seed is not None:
        random.seed(seed)

    profiles = []
    used_names: dict[str, set[str]] = {lang: set() for lang in LANGUAGES}
    used_projects: set[str] = set()
    used_classes: set[str] = set()

    idx = 0
    for lang in LANGUAGES:
        pool = PERSONA_POOLS[lang]
        for _ in range(users_per_lang):
            name = _pick_unique(pool, used_names[lang])

            prefix = _pick_unique(PROJECT_PREFIXES, used_projects)
            project_code = f"{prefix}-{random.randint(1000, 9999)}"

            cls_prefix = _pick_unique(CLASS_PREFIXES, used_classes)
            unique_class = f"{cls_prefix}Engine{random.randint(100, 999)}"

            const_type = random.choice(["RETRY", "BATCH", "QUEUE", "POOL", "CACHE"])

            profiles.append(UserProfile(
                user_id=f"user_{idx:03d}",
                language=lang,
                persona_name=name,
                project_code=project_code,
                canary_uuid=f"CANARY-{uuid.uuid4()}",
                canary_build=f"BUILD-{uuid.uuid4().hex[:12]}",
                unique_class=unique_class,
                unique_var=f"ctx_{uuid.uuid4().hex[:8]}",
                unique_const=f"MAX_{const_type}_{random.randint(1000, 9999)}",
            ))
            idx += 1

    return profiles


def _pick_unique(pool: list[str], used: set[str]) -> str:
    available = [x for x in pool if x not in used]
    if not available:
        base = random.choice(pool)
        variant = f"{base}_{random.randint(100, 999)}"
        used.add(variant)
        return variant
    choice = random.choice(available)
    used.add(choice)
    return choice
