import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from groq import Groq
from html import escape
import webbrowser  

# ---------- Conversor Markdown -> HTML (com fallback) ----------
try:
    import markdown as _mdlib
    def md_to_html(md_text: str) -> str:
        # Extensões úteis: tabelas, listas, quebras de linha
        return _mdlib.markdown(md_text, extensions=["extra", "tables", "sane_lists", "nl2br"])
except Exception:
    try:
        import markdown2 as _md2
        def md_to_html(md_text: str) -> str:
            return _md2.markdown(md_text, extras=["tables", "fenced-code-blocks"])
    except Exception:
        def md_to_html(md_text: str) -> str:
            # Fallback simples: exibe o Markdown “cru”
            return "<pre>" + escape(md_text) + "</pre>"

# Guarda o último relatório do Groq (em Markdown)
last_groq_md: Optional[str] = None

# =====================================================
# Config da API (cuidado ao versionar este arquivo)
# =====================================================
GROQ_API_KEY_INLINE = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY_INLINE

# guarda a última previsão feita (preenchida na opção 1)
last_pred: Optional[dict] = None


# =========================
# 1) Dataset de treino
# =========================
def montar_dataset() -> pd.DataFrame:
    cols = [
        "indice_vegetacao",
        "temperatura",
        "umidade",
        "vento_kmh",
        "chuva_7d_mm",
        "dias_sem_chuva",
        "queimada",
    ]
    dados = [
        (0.80, 26, 78, 10, 25, 0, 0),
        (0.72, 28, 70, 12, 18, 1, 0),
        (0.68, 29, 75, 8,  30, 0, 0),
        (0.90, 24, 85, 5,  40, 0, 0),
        (0.65, 27, 72, 15, 22, 2, 0),
        (0.70, 30, 68, 18, 12, 3, 0),
        (0.20, 43, 22, 38, 0,  10, 1),
        (0.15, 45, 18, 42, 2,  12, 1),
        (0.30, 41, 28, 35, 1,  9,  1),
        (0.25, 44, 25, 48, 0,  14, 1),
        (0.22, 42, 30, 33, 0,  11, 1),
        (0.40, 35, 45, 25, 5,  6,  1),
        (0.45, 33, 55, 20, 8,  5,  0),
        (0.55, 34, 50, 22, 6,  5,  0),
        (0.38, 36, 42, 28, 3,  8,  1),
        (0.60, 31, 60, 18, 10, 3,  0),
        (0.50, 32, 58, 16, 9,  4,  0),
        (0.35, 37, 40, 27, 4,  7,  1),
        (0.28, 39, 35, 30, 2,  9,  1),
        (0.62, 29, 68, 28, 20, 1, 0),
        (0.66, 30, 70, 26, 22, 0, 0),
        (0.48, 38, 38, 32, 1,  10, 1),
        (0.52, 40, 33, 34, 0,  12, 1),
    ]
    return pd.DataFrame(dados, columns=cols)


# =========================
# 2) Pipeline (Scaler + LogReg)
# =========================
def treinar_modelo(df: pd.DataFrame) -> Tuple[Pipeline, float]:
    X = df.drop(columns=["queimada"])
    y = df["queimada"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))
    ])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    return pipe, auc


# =========================
# 3) Entrada segura
# =========================
@dataclass
class Campo:
    nome: str
    minimo: float
    maximo: float
    default: float
    dica: str

def pedir_float(c: Campo) -> float:
    while True:
        try:
            raw = input(
                f"- {c.nome} ({c.dica}) [{c.minimo}–{c.maximo}] "
                f"(Enter p/ {c.default}): "
            ).strip()
            if raw == "":
                val = c.default
            else:
                val = float(raw.replace(",", "."))
            if not (c.minimo <= val <= c.maximo):
                print(f"  Valor fora do intervalo permitido [{c.minimo}–{c.maximo}].")
                continue
            return val
        except ValueError:
            print("  Entrada inválida. Digite um número (use . ou , para decimais).")

def coletar_entrada_usuario() -> pd.DataFrame:
    campos: Dict[str, Campo] = {
        "indice_vegetacao": Campo("Índice de vegetação (NDVI aprox.)", 0.0, 1.0, 0.50, "0=vegetação seca, 1=saudável"),
        "temperatura":      Campo("Temperatura (°C)",                 10.0, 55.0, 34.0, "ar/solo próximo à superfície"),
        "umidade":          Campo("Umidade relativa (%)",              5.0, 100.0, 50.0, "quanto menor, mais seco"),
        "vento_kmh":        Campo("Vento (km/h)",                      0.0, 120.0, 18.0, "ventos fortes espalham fogo"),
        "chuva_7d_mm":      Campo("Chuva 7 dias (mm)",                 0.0, 300.0, 8.0, "acumulado recente"),
        "dias_sem_chuva":   Campo("Dias sem chuva (dias)",             0.0, 60.0, 5.0, "acúmulo de dias secos"),
    }
    registro = {}
    print("\nInforme os dados (Enter usa o valor padrão sugerido):")
    for chave, cfg in campos.items():
        registro[chave] = pedir_float(cfg)
    return pd.DataFrame([registro])


# =========================
# 4) Previsão
# =========================
def classificar_risco(prob: float) -> str:
    if prob < 0.33:
        return "Baixo"
    elif prob < 0.66:
        return "Médio"
    else:
        return "Alto"

def prever_instancia(pipe: Pipeline, xnew: pd.DataFrame, limiar: float = 0.5) -> Tuple[int, float, str]:
    p = float(pipe.predict_proba(xnew)[0, 1])
    yhat = int(p >= limiar)
    return yhat, p, classificar_risco(p)


# =========================
# 5) Avaliação do modelo (texto)
# =========================
def _avaliar_modelo_texto(df: pd.DataFrame, pipe: Pipeline, alvo_recall: float = 0.85) -> Tuple[str, float]:
    X = df.drop(columns=["queimada"])
    y = df["queimada"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    proba = pipe.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)

    prec, rec, thr = precision_recall_curve(yte, proba)
    idx = next((i for i, r in enumerate(rec) if r >= alvo_recall), -1)
    limiar = thr[idx-1] if idx > 0 and (idx-1) < len(thr) else 0.5

    ypred_def = (proba >= 0.5).astype(int)
    ypred_rec = (proba >= limiar).astype(int)

    cm_def = confusion_matrix(yte, ypred_def)
    cm_rec = confusion_matrix(yte, ypred_rec)

    rep_def = classification_report(yte, ypred_def, digits=3)
    rep_rec = classification_report(yte, ypred_rec, digits=3)

    texto = []
    texto.append(f"ROC-AUC (hold-out): {auc:.3f}")
    texto.append("")
    texto.append("Matriz de confusão (limiar 0.50):")
    texto.append(str(cm_def))
    texto.append("Relatório (0.50):")
    texto.append(rep_def.strip())
    texto.append("")
    texto.append(f"Limiar escolhido para recall ≥ {alvo_recall:.2f}: {limiar:.3f}")
    texto.append("Matriz de confusão (limiar ajustado p/ recall):")
    texto.append(str(cm_rec))
    texto.append("Relatório (limiar ajustado):")
    texto.append(rep_rec.strip())
    return "\n".join(texto), limiar


def avaliar_modelo(df: pd.DataFrame, pipe: Pipeline) -> None:
    texto, _ = _avaliar_modelo_texto(df, pipe, alvo_recall=0.85)
    print("\n=== Avaliação do Modelo ===")
    print(texto)


# =========================
# 6) Visualização (dispersão)
# =========================
def plotar_dispersao(df: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(
        df["temperatura"], df["umidade"],
        c=df["queimada"], s=np.clip(df["vento_kmh"]*4, 20, 400),
        cmap="coolwarm", edgecolors="k", alpha=0.85
    )
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Umidade relativa (%)")
    ax.set_title("Distribuição (tamanho ~ vento; cor 0/1 = sem/queimada)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Classe (0=Sem Queimada, 1=Queimada)")
    plt.show()


# =========================
# 7) Importâncias (coeficientes)
# =========================
def mostrar_coeficientes(pipe: Pipeline, colunas: pd.Index) -> None:
    clf: LogisticRegression = pipe.named_steps["clf"]  # type: ignore
    coefs = clf.coef_[0]
    dfc = pd.DataFrame({"variavel": colunas, "coef": coefs}).sort_values("coef", ascending=False)
    print("\nCoeficientes (escala padronizada; positivos aumentam o risco):")
    print(dfc.to_string(index=False))


def _coeficientes_texto(pipe: Pipeline, colunas: pd.Index, k: int = 3) -> str:
    clf: LogisticRegression = pipe.named_steps["clf"]  # type: ignore
    coefs = clf.coef_[0]
    dfc = pd.DataFrame({"variavel": colunas, "coef": coefs}).sort_values("coef", ascending=False)
    top_pos = dfc.head(k).to_records(index=False).tolist()
    top_neg = dfc.tail(k).sort_values("coef").to_records(index=False).tolist()
    linhas = ["Top variáveis que aumentam risco:"]
    for v, c in top_pos:
        linhas.append(f"  + {v}: {c:.3f}")
    linhas.append("Top variáveis que reduzem risco:")
    for v, c in top_neg:
        linhas.append(f"  - {v}: {c:.3f}")
    return "\n".join(linhas)


# =========================
# 8) RELATÓRIO DA PREVISÃO (Groq) + fallback + export
# =========================
def _contribuicoes_instancia(pipe: Pipeline, xnew: pd.DataFrame) -> str:
    """Explicação linear aproximada: contrib = coef * zscore."""
    scaler: StandardScaler = pipe.named_steps["scaler"]  # type: ignore
    clf: LogisticRegression = pipe.named_steps["clf"]     # type: ignore

    cols = xnew.columns
    x = xnew.iloc[0].values.astype(float)
    z = (x - scaler.mean_) / scaler.scale_
    contrib = clf.coef_[0] * z

    dfc = pd.DataFrame({"variavel": cols, "zscore": z, "coef": clf.coef_[0], "contrib": contrib})
    dfc_sorted = dfc.sort_values("contrib", ascending=False)

    top_up = dfc_sorted.head(3)
    top_down = dfc_sorted.tail(3)

    linhas = ["Principais fatores que AUMENTARAM o risco:"]
    for _, r in top_up.iterrows():
        linhas.append(f"  + {r['variavel']}: contrib={r['contrib']:.3f} (z={r['zscore']:.2f}, coef={r['coef']:.3f})")

    linhas.append("Principais fatores que REDUZIRAM o risco:")
    for _, r in top_down.iterrows():
        linhas.append(f"  - {r['variavel']}: contrib={r['contrib']:.3f} (z={r['zscore']:.2f}, coef={r['coef']:.3f})")

    return "\n".join(linhas)


def _montar_prompt_relatorio_previsao(case: dict, pipe: Pipeline) -> str:
    xdf: pd.DataFrame = case["input"]
    vals = xdf.to_dict(orient="records")[0]
    feat_lines = "\n".join([f"| **{k}** | {v} |" for k, v in vals.items()])

    contrib_txt = _contribuicoes_instancia(pipe, xdf)

    prob = case["prob"]
    risco = case["risk"]
    status = case["class"]
    limiar = case["threshold"]
    quando = case["time"]

    # badge por nível de risco
    badge = "🟢 Baixo" if prob < 0.33 else ("🟠 Médio" if prob < 0.66 else "🔴 Alto")

    return f"""
Formate o relatório **em Markdown elegante**, com:
- Títulos `##` e `###`
- Uma linha de **badges** (emoji do nível de risco)
- Tabela das variáveis
- Bullets claros e curtos
- Separadores `---`

Use *apenas* os números fornecidos. Não invente nada.

## 🔥 Relatório de Risco de Queimada — Previsão Individual

**Data/hora:** {quando}  
**Limiar usado:** {limiar:.2f}  
**Probabilidade estimada de queimada: ** **{prob:.3f}**  
**Nível de risco:** **{badge}**  
**Classe prevista:** **{status}**

---

### 1) Resumo
Escreva 2–4 frases objetivas explicando o que significa essa probabilidade e como agir em termos gerais.

---

### 2) Detalhes da Amostra
| Variável | Valor |
|---------:|------:|
{feat_lines}

> Observação: valores são padronizados internamente antes da predição.

---

### 3) Interpretação (efeitos aproximados)
Explique, em bullets, os fatores que **aumentaram** e que **reduziram** o risco, usando as contribuições abaixo como guia (sem reescrever tudo, resuma com clareza):

{contrib_txt}

---

### 4) Recomendações Operacionais
Escreva 3 recomendações práticas coerentes com **{badge}**.  
Ex.: fiscalização, comunicação preventiva, restrições temporárias, priorização por local/horário/vento.

---

### 5) Observações
- Explicação linear e aproximada com base em coeficientes padronizados do modelo.
- Adapte o limiar de alerta conforme a política local de custos/erros.
""".strip()


def _risk_badge_and_color(prob: float) -> Tuple[str, str]:
    if prob >= 0.66:
        return "🔴 Alto", "#ef4444"
    elif prob >= 0.33:
        return "🟠 Médio", "#f59e0b"
    return "🟢 Baixo", "#22c55e"


def salvar_relatorio_previsao_md(case: Optional[dict], pipe: Pipeline, path: Optional[str] = None) -> None:
    if not case:
        print("\n[Relatório] Nenhuma previsão encontrada. Faça a opção 1 primeiro.\n")
        return
    xdf: pd.DataFrame = case["input"]
    vals = xdf.to_dict(orient="records")[0]
    rows = "\n".join([f"| **{k}** | {v} |" for k, v in vals.items()])
    badge, _ = _risk_badge_and_color(case["prob"])
    md = f"""# 🔥 Relatório de Risco de Queimada — Previsão Individual

**Data/hora:** {case['time']}  
**Limiar usado:** {case['threshold']:.2f}  
**Probabilidade estimada de queimada:** **{case['prob']:.3f}**  
**Nível de risco:** **{badge}**  
**Classe prevista:** **{case['class']}**

---

## Resumo
- Risco **{badge}** com probabilidade de {case['prob']:.1%}.  
- Interprete conforme políticas locais de alarme e custo de erro.

---

## Detalhes da Amostra
| Variável | Valor |
|---------:|------:|
{rows}

> Observação: os valores são padronizados internamente antes da predição.

---

## Interpretação
{_contribuicoes_instancia(pipe, xdf)}

---

## Recomendações Operacionais
{"- Acionar alerta vermelho e priorizar fiscalização; mobilizar brigada; restringir atividades de risco." if case["prob"]>=0.66 else ("- Alerta amarelo: intensificar vigilância; notificar limpeza de aceiros; planejar rondas em horários críticos." if case["prob"]>=0.33 else "- Monitoramento de rotina; comunicação preventiva; acompanhar vento e umidade.")}

---

## Observações
- Explicação linear aproximada (coeficientes padronizados).
- Ajuste o limiar conforme a política local.
"""
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"relatorio_previsao_{stamp}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Relatório salvo em Markdown: {path}")


import webbrowser  # <-- 


def salvar_relatorio_previsao_html(
    case: Optional[dict],
    pipe: Pipeline,
    path: Optional[str] = None,
    groq_md: Optional[str] = None
) -> None:
    if not case:
        print("\n[Relatório] Nenhuma previsão encontrada. Faça a opção 1 primeiro.\n")
        return
    xdf: pd.DataFrame = case["input"]
    vals = xdf.to_dict(orient="records")[0]
    rows = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in vals.items()])

    badge, color = _risk_badge_and_color(case["prob"])
    pct = f"{case['prob']*100:.1f}%"
    interp = _contribuicoes_instancia(pipe, xdf).replace("\n", "<br>")

    # Se vier Markdown do Groq, converte para HTML
    groq_html = md_to_html(groq_md) if groq_md else ""

    html = f"""<!DOCTYPE html>
<html lang="pt-br"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de Risco – Previsão</title>
<style>
  :root {{ --accent: {color}; --bg: #0b1020; --fg: #e5e7eb; --muted: #9ca3af; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
         background: var(--bg); color: var(--fg); margin: 0; padding: 32px; }}
  .card {{ max-width: 900px; margin: 0 auto; background: #0f172a; border: 1px solid #1f2937;
           border-radius: 16px; padding: 28px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }}
  h1 {{ margin: 0 0 8px; font-size: 28px; }}
  .meta {{ color: var(--muted); margin-bottom: 18px; }}
  .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: var(--accent);
            color:#0b1020; font-weight: 700; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 22px 0; }}
  table {{ width:100%; border-collapse: collapse; }}
  td, th {{ padding: 8px 10px; border-bottom: 1px solid #273244; }}
  th {{ text-align: left; color: var(--muted); font-weight: 600; }}
  .bar {{ height: 14px; background:#1f2a3c; border-radius: 999px; overflow: hidden; border:1px solid #273244; }}
  .bar > span {{ display:block; height:100%; width:{pct}; background: var(--accent); }}
  .section {{ margin-top: 24px; }}
  .callout {{ border-left: 4px solid var(--accent); padding: 10px 12px; background:#0b1426; border-radius: 8px; }}
  code {{ background:#0b1426; padding:2px 6px; border-radius:6px; border:1px solid #1f2937; }}

  .md h2, .md h3 {{ margin: 16px 0 8px; }}
  .md table {{ width:100%; border-collapse: collapse; margin: 12px 0; }}
  .md th, .md td {{ border: 1px solid #273244; padding: 6px 8px; }}
  .md blockquote {{ border-left: 4px solid var(--accent); padding: 8px 12px; background:#0b1426; border-radius: 8px; }}
  .md code {{ background:#0b1426; padding:2px 6px; border-radius:6px; border:1px solid #1f2937; }}
  .muted {{ color: var(--muted); }}
</style></head>
<body>
  <main class="card">
    <h1>🔥 Relatório de Risco de Queimada — Previsão Individual</h1>
    <div class="meta">Data/hora: {case['time']} • Limiar: <code>{case['threshold']:.2f}</code></div>

    <div class="grid">
      <div>
        <div class="section">
          <div class="badge">{badge}</div>
        </div>
        <div class="section">
          <div class="callout"><strong>Probabilidade estimada:</strong> {pct}</div>
        </div>
        <div class="section">
          <div class="bar"><span></span></div>
        </div>
      </div>
      <div>
        <table>
          <thead><tr><th>Variável</th><th>Valor</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>Interpretação</h2>
      <p>{interp}</p>
    </div>

    <div class="section">
      <h2>Recomendações Operacionais</h2>
      <ul>
        {"<li>Acionar alerta vermelho e priorizar fiscalização; mobilizar brigada; restringir atividades de risco.</li>" if case["prob"]>=0.66 else (
          "<li>Alerta amarelo: intensificar vigilância; notificar limpeza de aceiros; planejar rondas em horários críticos.</li>" if case["prob"]>=0.33 else
          "<li>Monitoramento de rotina; comunicação preventiva; acompanhar vento e umidade.</li>"
        )}
      </ul>
    </div>

    {"<div class='section'><h2>Relatório (Groq)</h2><div class='md'>" + groq_html + "</div></div>" if groq_html else ""}

    <div class="section">
      <h2>Observações</h2>
      <p class="muted">Explicação linear aproximada (coeficientes padronizados). Ajuste o limiar conforme a política local.</p>
    </div>
  </main>
</body></html>
"""
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"relatorio_previsao_{stamp}.html"

    # Salva o HTML
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Relatório salvo em HTML: {path}")

    # 🔥 abre automaticamente no navegador
    webbrowser.open_new_tab("file://" + os.path.abspath(path))



# =========================
# Função para gerar relatório Groq (opção 6)
# =========================
def gerar_relatorio_groq_previsao(case: Optional[dict], pipe: Pipeline) -> Optional[str]:
    global last_groq_md
    if not case:
        print("\n[Groq] Nenhuma previsão encontrada. Faça a opção 1 primeiro.\n")
        return None
    prompt = _montar_prompt_relatorio_previsao(case, pipe)
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "Você é um especialista em incêndios florestais e análise de risco. Responda em Markdown elegante, claro e objetivo."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.2,
            stream=False,
        )
        md = (response.choices[0].message.content or "").strip()
        last_groq_md = md
        print("\n=== Relatório Groq (Markdown) ===\n")
        print(md)
        return md
    except Exception as e:
        print(f"[Groq] Falha ao gerar relatório: {e}")
        last_groq_md = None
        return None


# =========================
# 9) Loop interativo
# =========================
def main():
    global last_pred, last_groq_md  # vamos salvar a última previsão e o último relatório Groq

    df = montar_dataset()
    pipe, auc = treinar_modelo(df)
    print("Modelo treinado com sucesso! ROC-AUC (validação hold-out) =", f"{auc:.3f}")

    limiar = 0.5

    while True:
        print("\n=== Menu ===")
        print("1) Fazer previsão manual (digitar variáveis)")
        print("2) Avaliar modelo (métricas e limiar focado em recall)")
        print("3) Visualizar dados (dispersão temperatura x umidade)")
        print("4) Ver coeficientes (importância das variáveis)")
        print("5) Ajustar limiar de alerta (padrão atual: %.2f)" % limiar)
        print("6) Gerar relatório da PREVISÃO (Groq)")
        print("7) Salvar relatório da PREVISÃO em Markdown (.md)")
        print("8) Salvar relatório da PREVISÃO em HTML (.html)")
        print("0) Sair")
        op = input("Escolha uma opção: ").strip()

        if op == "1":
            xnew = coletar_entrada_usuario()
            yhat, p, risco = prever_instancia(pipe, xnew, limiar=limiar)
            status = "QUEIMADA" if yhat == 1 else "SEM QUEIMADA"
            print("\n=== Resultado ===")
            print(xnew.to_string(index=False))
            print(f"Probabilidade estimada: {p:.3f}  |  Risco: {risco}  |  Classe: {status}")

            # salvar a última previsão para o relatório
            last_pred = {
                "input": xnew,
                "prob": p,
                "risk": risco,
                "class": status,
                "threshold": limiar,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        elif op == "2":
            avaliar_modelo(df, pipe)

        elif op == "3":
            try:
                plotar_dispersao(df)
            except Exception as e:
                print(f"Falha ao plotar: {e}")

        elif op == "4":
            mostrar_coeficientes(pipe, df.drop(columns=['queimada']).columns)

        elif op == "5":
            try:
                novo = input("Novo limiar (0–1). Ex.: 0.40 para mais sensível: ").strip()
                if novo == "":
                    print("Limiar mantido.")
                else:
                    val = float(novo.replace(",", "."))
                    if 0.0 <= val <= 1.0:
                        limiar = val
                        print(f"Limiar atualizado para {limiar:.2f}.")
                    else:
                        print("Valor fora do intervalo [0–1].")
            except ValueError:
                print("Valor inválido. Tente novamente.")

        elif op == "6":
            # Gera e guarda o Markdown do Groq (também imprime no console)
            gerar_relatorio_groq_previsao(last_pred, pipe)

        elif op == "7":
            salvar_relatorio_previsao_md(last_pred, pipe)

        elif op == "8":
            # Salva o HTML, injetando o markdown do Groq (se existir)
            salvar_relatorio_previsao_html(last_pred, pipe, groq_md=last_groq_md)

        elif op == "0":
            print("Encerrando. Até mais!")
            break
        else:
            print("Opção inválida. Tente novamente.")


# =========================
# Único bloco principal
# =========================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(0)
