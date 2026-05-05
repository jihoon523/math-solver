from fastapi import FastAPI, File, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sympy import (
    symbols, Eq, solve, sympify, diff, integrate,
    factor, expand, simplify
)
import re
import os
import httpx
import base64
import sqlite3
import datetime
from typing import Optional
from dotenv import load_dotenv
import jwt
from passlib.context import CryptContext

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_INPUT_LENGTH = 2000
SECRET_KEY = os.environ.get("SECRET_KEY", "mathsolver-secret-key-2024")
WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
DB_PATH = "users.db"


# ── 데이터베이스 초기화 ─────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            problem TEXT NOT NULL,
            problem_type TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        return None


# ── 요청 모델 ────────────────────────────────────────────────
class SolveRequest(BaseModel):
    problem: str


class AuthRequest(BaseModel):
    username: str
    password: str


class SaveHistoryRequest(BaseModel):
    problem: str
    problem_type: str
    answer: str


# ── 텍스트 파싱 ──────────────────────────────────────────────
def parse_structured(text: str) -> dict:
    def extract(label, content):
        m = re.search(rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)", content, re.DOTALL)
        return m.group(1).strip() if m else ""

    solution_m = re.search(r"SOLUTION:\s*\n(.*?)(?=\nEXPLANATION:|$)", text, re.DOTALL)
    return {
        "problem_type": extract("PROBLEM_TYPE", text),
        "answer":        extract("ANSWER", text),
        "solution":      solution_m.group(1).strip() if solution_m else text,
        "explanation":   extract("EXPLANATION", text),
    }


def detect_variable(problem: str):
    for v in ["x", "y", "z", "t", "n", "k"]:
        if re.search(rf'\b{v}\b', problem):
            return symbols(v)
    return symbols("x")


def normalize(problem: str) -> str:
    remove_words = [
        "을 풀어줘", "를 풀어줘", "풀어줘", "풀어라", "풀어",
        "구해줘", "구해라", "구해", "해결해줘", "계산해줘",
        "미분해줘", "미분해", "적분해줘", "적분해",
        "인수분해해줘", "인수분해해", "인수분해",
        "전개해줘", "전개해", "단순화해줘",
    ]
    for word in remove_words:
        problem = problem.replace(word, "")
    problem = problem.strip()
    problem = problem.replace(" ", "").replace("^", "**")
    problem = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", problem)
    problem = re.sub(r"(\))(\()", r"\1*\2", problem)
    problem = re.sub(r"(\d)(\()", r"\1*\2", problem)
    problem = re.sub(r"(\))(\d)", r"\1*\2", problem)
    return problem


def detect_mode(original: str) -> str:
    if any(k in original for k in ["미분", "diff", "d/dx"]):
        return "diff"
    if any(k in original for k in ["적분", "integrate", "∫"]):
        return "integrate"
    if any(k in original for k in ["인수분해", "factor"]):
        return "factor"
    if any(k in original for k in ["전개", "expand"]):
        return "expand"
    if "=" in original:
        return "equation"
    return "simplify"


def sympy_solve(problem: str, original: str) -> dict:
    var = detect_variable(original)
    var_name = str(var)
    mode = detect_mode(original)

    try:
        if mode == "equation":
            left, right = problem.split("=", 1)
            eq = Eq(sympify(left), sympify(right))
            solution = solve(eq, var)
            if not solution:
                return {"success": False, "error": "SymPy가 해를 찾지 못했습니다."}
            answers = [f"{var_name} = {s}" for s in solution]
            return {
                "success": True, "mode": "equation",
                "parsed": str(eq), "answer": ", ".join(answers),
            }
        elif mode == "diff":
            expr = sympify(problem)
            result = diff(expr, var)
            return {"success": True, "mode": "diff", "parsed": str(expr), "answer": str(result)}
        elif mode == "integrate":
            expr = sympify(problem)
            result = integrate(expr, var)
            return {"success": True, "mode": "integrate", "parsed": str(expr), "answer": str(result) + " + C"}
        elif mode == "factor":
            expr = sympify(problem)
            result = factor(expr)
            return {"success": True, "mode": "factor", "parsed": str(expr), "answer": str(result)}
        elif mode == "expand":
            expr = sympify(problem)
            result = expand(expr)
            return {"success": True, "mode": "expand", "parsed": str(expr), "answer": str(result)}
        else:
            expr = sympify(problem)
            result = simplify(expr)
            if str(result) == str(expr):
                return {"success": False, "error": "SymPy로 단순화할 수 없습니다."}
            return {"success": True, "mode": "simplify", "parsed": str(expr), "answer": str(result)}
    except Exception as e:
        return {"success": False, "error": f"계산 오류: {str(e)}"}


async def _call_claude(messages: list, max_tokens: int = 1200) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


async def claude_explain(original_problem: str, sympy_result: dict) -> dict:
    mode_labels = {
        "equation": "방정식", "diff": "미분", "integrate": "부정적분",
        "factor": "인수분해", "expand": "전개", "simplify": "단순화",
    }
    if not CLAUDE_API_KEY:
        return {
            "problem_type": mode_labels.get(sympy_result.get("mode", ""), "수식"),
            "solution": f"**풀이**\n\n입력된 식: ${sympy_result['parsed']}$\n\n**결과:** ${sympy_result['answer']}$",
            "answer": sympy_result["answer"],
            "explanation": "Claude API 키가 설정되지 않아 기본 풀이만 제공됩니다.",
        }

    prompt = f"""다음 수학 문제와 계산 결과를 바탕으로 중학생도 이해할 수 있게 단계별로 설명해주세요.

원본 문제: {original_problem}
계산 결과: {sympy_result['answer']}
계산 모드: {sympy_result['mode']}

아래 형식으로 정확히 답하세요:

PROBLEM_TYPE: (문제 유형, 예: 일차방정식)
ANSWER: (최종 정답)
SOLUTION:
(마크다운 형식으로 단계별 풀이. 수식은 인라인 $...$, 블록 $$...$$로 표시. 각 단계는 **굵게** 소제목 사용. 최소 4단계)
EXPLANATION: (핵심 포인트 한 문장)"""

    text = await _call_claude([{"role": "user", "content": prompt}])
    return parse_structured(text)


def extract_polynomial_roots(problem: str):
    """문제에서 다항식을 추출하고 근을 수치로 계산"""
    text = problem
    for sup, exp in [("³","**3"),("²","**2"),("¹","**1"),("^3","**3"),("^2","**2"),("^","**")]:
        text = text.replace(sup, exp)
    text = re.sub(r'(\d)([x])', r'\1*\2', text)

    match = re.search(r'([x\d\s\+\-\*\(\)\.\*]+)\s*=\s*0', text)
    if not match:
        return None
    poly_str = match.group(1).strip()
    try:
        x = symbols('x')
        poly = sympify(poly_str)
        roots = solve(poly, x)
        real_roots = []
        for r in roots:
            try:
                val = float(r.evalf())
                real_roots.append(round(val, 10))
            except Exception:
                pass
        return real_roots if real_roots else None
    except Exception:
        return None


def math_to_python(expr: str) -> str:
    """수학 표기를 Python 실행 가능한 코드로 변환"""
    # 그리스 문자 → 변수명
    expr = expr.replace('α', 'a').replace('β', 'b').replace('γ', 'c')
    expr = expr.replace('[', '(').replace(']', ')')
    # 유니코드 위첨자 및 ^ 표기
    for sup, exp in [
        ("⁹","**9"),("⁸","**8"),("⁷","**7"),("⁶","**6"),("⁵","**5"),("⁴","**4"),
        ("³","**3"),("²","**2"),("¹","**1"),
        ("^9","**9"),("^8","**8"),("^7","**7"),("^6","**6"),("^5","**5"),("^4","**4"),
        ("^3","**3"),("^2","**2"),("^","**"),
    ]:
        expr = expr.replace(sup, exp)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'\)\s*\(', r')*(', expr)
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
    return expr.strip()


def extract_math_expression(problem: str) -> str:
    """문제에서 계산할 수식 부분만 추출"""
    # 한국어 패턴: 수식이 "의 값을 구하시오" 앞에 위치
    for end_marker in ["의 값을 구하시오", "의값을구하시오", "의 값을 구하여라", "의 값을구하시오"]:
        idx = problem.find(end_marker)
        if idx != -1:
            before = problem[:idx].strip()
            # "라 할 때" 이후 부분이 수식
            for delimiter in ["라 할 때,", "라 할 때", "라고 할 때,", "라고 할 때", "할 때,", "할 때"]:
                d_idx = before.rfind(delimiter)
                if d_idx != -1:
                    expr = before[d_idx + len(delimiter):].strip().lstrip(",").strip()
                    # "다음 식" 등 앞의 한국어 설명 제거
                    expr = re.sub(r'^[가-힣\s]+', '', expr).strip()
                    if expr:
                        return expr
            # 딜리미터 없으면 "= 0" 이후 부분
            if "= 0" in before:
                expr = before.split("= 0", 1)[-1].strip().lstrip(",").strip()
                expr = re.sub(r'^[가-힣\s]+', '', expr).strip()
                if expr:
                    return expr
            return before

    # 영어 패턴: 수식이 마커 뒤에 위치
    for marker in ["compute sum of", "compute", "find the value of", "evaluate"]:
        idx = problem.lower().find(marker.lower())
        if idx != -1:
            after = problem[idx + len(marker):].strip().lstrip(",").strip()
            if "= 0" in after:
                after = after.split("= 0", 1)[-1].strip().lstrip(",").strip()
            return after
    return ""


async def numeric_root_evaluation(problem: str) -> dict:
    """다항식 근을 수치 계산하고 수식을 직접 Python으로 변환해 실행"""
    roots = extract_polynomial_roots(problem)
    if not roots or len(roots) < 2:
        return {"success": False}

    a = roots[0]
    b = roots[1]
    c = roots[2] if len(roots) > 2 else 0.0

    # 1차 시도: 문제에서 직접 수식 추출
    raw_expr = extract_math_expression(problem)
    expr_code = math_to_python(raw_expr) if raw_expr else ""

    # 2차 시도: Claude에게 코드 생성 요청
    if not expr_code:
        prompt = f"""Convert this math problem's expression to a single Python arithmetic expression.
Roots already computed: a={a}, b={b}, c={c}
Problem: {problem}
Rules: use only a,b,c variables, **, *, +, -, /, (, ) operators only. No imports or functions.
Output only: EXPRESSION: <python expression>"""
        try:
            code_text = await _call_claude([{"role": "user", "content": prompt}], max_tokens=400)
            for line in code_text.splitlines():
                if "EXPRESSION:" in line:
                    expr_code = line.split("EXPRESSION:", 1)[1].strip().strip("`")
                    break
            if not expr_code and code_text.strip():
                expr_code = code_text.strip().splitlines()[-1].strip("`")
        except Exception:
            pass

    if not expr_code:
        return {"success": False}

    if re.search(r'(import|exec|eval|open|os|sys|__)', expr_code):
        return {"success": False}

    # 순환합 여부 판단 ("sum of" 또는 "∑" 포함 시)
    is_cyclic = "sum of" in problem.lower() or "∑" in problem or "순환합" in problem

    try:
        safe_ns = {"__builtins__": {}}
        if is_cyclic:
            # 순환합: f(a,b,c) + f(b,c,a) + f(c,a,b)
            r1 = eval(expr_code, safe_ns, {"a": a, "b": b, "c": c})
            r2 = eval(expr_code, safe_ns, {"a": b, "b": c, "c": a})
            r3 = eval(expr_code, safe_ns, {"a": c, "b": a, "c": b})
            result = float(r1) + float(r2) + float(r3)
            calc_note = f"순환합 = f(a,b,c) + f(b,c,a) + f(c,a,b)\n= {round(float(r1),6)} + {round(float(r2),6)} + {round(float(r3),6)}"
        else:
            result = float(eval(expr_code, safe_ns, {"a": a, "b": b, "c": c}))
            calc_note = f"계산식: `{expr_code}`"

        if abs(result - round(result)) < 0.001:
            result = int(round(result))

        return {
            "success": True,
            "problem_type": "대칭식 (수치 계산)",
            "answer": str(result),
            "solution": (
                f"**다항식의 근 (수치)**\n\n"
                f"$$a = {round(a,6)}, \\quad b = {round(b,6)}, \\quad c = {round(c,6)}$$\n\n"
                f"**각 항의 수식**\n\n`{expr_code}`\n\n"
                f"**{calc_note}**\n\n"
                f"**결과:** ${result}$"
            ),
            "explanation": "SymPy로 근을 수치 계산한 뒤 Python으로 직접 연산한 정확한 결과입니다.",
        }
    except Exception:
        return {"success": False}


async def wolfram_solve(problem: str) -> dict:
    if not WOLFRAM_APP_ID:
        return {"success": False}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.wolframalpha.com/v2/query",
                params={
                    "appid": WOLFRAM_APP_ID,
                    "input": problem,
                    "output": "json",
                    "format": "plaintext",
                    "podstate": "Result__Step-by-step solution",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("queryresult", {}).get("success"):
            return {"success": False}

        pods = data["queryresult"].get("pods", [])
        answer = ""
        steps = []

        for pod in pods:
            title = pod.get("title", "")
            plaintext = pod.get("subpods", [{}])[0].get("plaintext", "")
            if not plaintext:
                continue
            if title in ("Result", "Exact result", "Value", "Solution", "Results"):
                answer = plaintext
            if title not in ("Input", "Input interpretation"):
                steps.append(f"**{title}**\n{plaintext}")

        if not answer and steps:
            answer = steps[0].split("\n", 1)[-1].strip()

        if not answer:
            return {"success": False}

        solution = "\n\n".join(steps)
        return {
            "success": True,
            "problem_type": "Wolfram Alpha 계산",
            "answer": answer,
            "solution": solution,
            "explanation": "Wolfram Alpha 엔진으로 계산된 결과입니다.",
        }
    except Exception:
        return {"success": False}


async def claude_solve_directly(problem: str) -> dict:
    if not CLAUDE_API_KEY:
        return {"success": False, "error": "Claude API 키가 설정되지 않았습니다."}

    prompt = f"""다음 수학 문제를 중학생도 이해할 수 있도록 쉽고 친절하게 단계별로 풀어주세요.

문제: {problem}

아래 형식으로 정확히 답하세요:

PROBLEM_TYPE: (문제 유형, 예: 수열, 함수, 확률 등)
ANSWER: (최종 정답)
SOLUTION:
(마크다운 형식으로 단계별 풀이. 수식은 인라인 $...$, 블록 $$...$$로 표시. 각 단계는 **굵게** 소제목으로 구분. 구체적인 숫자 계산 포함. 최소 5단계)
EXPLANATION: (이 문제의 핵심 포인트 한 문장)"""

    text = await _call_claude([{"role": "user", "content": prompt}], max_tokens=1500)
    parsed = parse_structured(text)
    return {"success": True, **parsed}


# ── 인증 엔드포인트 ──────────────────────────────────────────
@app.post("/register")
async def register(request: AuthRequest):
    if len(request.username) < 3:
        return {"success": False, "error": "아이디는 3자 이상이어야 합니다."}
    if len(request.password) < 6:
        return {"success": False, "error": "비밀번호는 6자 이상이어야 합니다."}
    conn = get_db()
    try:
        password_hash = pwd_context.hash(request.password)
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (request.username, password_hash)
        )
        conn.commit()
        user = conn.execute("SELECT id FROM users WHERE username = ?", (request.username,)).fetchone()
        token = create_token(user["id"], request.username)
        return {"success": True, "token": token, "username": request.username}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "이미 사용 중인 아이디입니다."}
    finally:
        conn.close()


@app.post("/login")
async def login(request: AuthRequest):
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (request.username,)).fetchone()
        if not user or not pwd_context.verify(request.password, user["password_hash"]):
            return {"success": False, "error": "아이디 또는 비밀번호가 틀렸습니다."}
        token = create_token(user["id"], request.username)
        return {"success": True, "token": token, "username": request.username}
    finally:
        conn.close()


@app.get("/history")
async def get_history(user=Depends(get_current_user)):
    if not user:
        return {"success": False, "error": "로그인이 필요합니다."}
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, problem, problem_type, answer, created_at FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
            (user["user_id"],)
        ).fetchall()
        return {"success": True, "history": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.post("/history/save")
async def save_history_endpoint(request: SaveHistoryRequest, user=Depends(get_current_user)):
    if not user:
        return {"success": False}
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO history (user_id, problem, problem_type, answer) VALUES (?, ?, ?, ?)",
            (user["user_id"], request.problem, request.problem_type, request.answer)
        )
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


# ── 기존 엔드포인트 ──────────────────────────────────────────
@app.post("/solve")
async def solve_problem(request: SolveRequest):
    original = request.problem.strip()

    if not original:
        return {"success": False, "error": "문제를 입력해주세요."}
    if len(original) > MAX_INPUT_LENGTH:
        return {"success": False, "error": f"입력이 너무 깁니다. (최대 {MAX_INPUT_LENGTH}자)"}

    normalized = normalize(original)
    sympy_result = sympy_solve(normalized, original)

    if not sympy_result["success"]:
        # 수치 근 계산 시도 (대칭식 문제)
        numeric_result = await numeric_root_evaluation(original)
        if numeric_result["success"]:
            return {
                "success": True,
                "problem_type": numeric_result.get("problem_type", "수학 문제"),
                "parsed_expression": original,
                "answer": numeric_result.get("answer", ""),
                "solution": numeric_result.get("solution", ""),
                "explanation": numeric_result.get("explanation", ""),
                "verified": True,
            }
        # Wolfram Alpha 시도
        wolfram_result = await wolfram_solve(original)
        if wolfram_result["success"]:
            return {
                "success": True,
                "problem_type": wolfram_result.get("problem_type", "수학 문제"),
                "parsed_expression": original,
                "answer": wolfram_result.get("answer", ""),
                "solution": wolfram_result.get("solution", ""),
                "explanation": wolfram_result.get("explanation", ""),
                "verified": True,
            }
        # Claude 시도
        try:
            result = await claude_solve_directly(original)
            if result["success"]:
                return {
                    "success": True,
                    "problem_type": result.get("problem_type", "수학 문제"),
                    "parsed_expression": original,
                    "answer": result.get("answer", ""),
                    "solution": result.get("solution", ""),
                    "explanation": result.get("explanation", ""),
                    "verified": True,
                }
        except Exception:
            pass
        return {"success": False, "error": sympy_result["error"]}

    try:
        claude_result = await claude_explain(original, sympy_result)
    except Exception as e:
        claude_result = {
            "problem_type": sympy_result.get("mode", "계산"),
            "solution": f"**결과:** {sympy_result['answer']}",
            "answer": sympy_result["answer"],
            "explanation": f"Claude 설명 오류: {str(e)}",
        }

    return {
        "success": True,
        "problem_type": claude_result.get("problem_type", sympy_result["mode"]),
        "parsed_expression": sympy_result["parsed"],
        "answer": claude_result.get("answer", sympy_result["answer"]),
        "solution": claude_result.get("solution", ""),
        "explanation": claude_result.get("explanation", ""),
        "verified": True,
    }


@app.post("/hint")
async def get_hint(request: SolveRequest):
    problem = request.problem.strip()
    if not problem:
        return {"success": False, "hint": "문제를 먼저 입력해주세요."}
    if not CLAUDE_API_KEY:
        return {"success": False, "hint": "힌트를 생성하려면 Claude API 키가 필요합니다."}

    prompt = f"""다음 수학 문제에 대한 힌트를 하나만 제공해주세요.
정답은 절대 말하지 말고, 풀이 방향만 살짝 알려주세요. 두 문장 이내로 간결하게.

문제: {problem}

힌트:"""

    try:
        text = await _call_claude([{"role": "user", "content": prompt}], max_tokens=200)
        return {"success": True, "hint": text}
    except Exception as e:
        return {"success": False, "hint": f"힌트 생성 오류: {str(e)}"}


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"success": False, "error": "PDF 파일만 지원합니다."}
    if not CLAUDE_API_KEY:
        return {"success": False, "error": "PDF 인식을 위해 ANTHROPIC_API_KEY 환경변수를 설정해주세요."}

    try:
        pdf_bytes = await file.read()
        b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
        text = await _call_claude([{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                {"type": "text", "text": "이 PDF에 있는 수학 문제를 텍스트로만 추출해주세요. 수식 기호는 그대로 유지하고, 문제 텍스트만 출력하세요. 설명 없이 문제만 출력하세요."}
            ],
        }], max_tokens=500)
        return {"success": True, "problem": text}
    except Exception as e:
        return {"success": False, "error": f"PDF 처리 오류: {str(e)}"}


@app.post("/extract-image")
async def extract_image(file: UploadFile = File(...)):
    allowed = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed:
        return {"success": False, "error": "지원하지 않는 이미지 형식입니다. (jpg, png, gif, webp)"}

    try:
        image_bytes = await file.read()
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        text = await _call_claude([{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": file.content_type, "data": b64}},
                {"type": "text", "text": "이 이미지에 있는 수학 문제를 텍스트로만 추출해주세요. 수식 기호는 그대로 유지하고, 문제 텍스트만 출력하세요. 설명 없이 문제만 출력하세요."}
            ],
        }], max_tokens=500)
        return {"success": True, "problem": text}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"이미지 처리 오류: {str(e)}"}
