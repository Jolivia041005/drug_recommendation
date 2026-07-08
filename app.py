import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from drug_recommendation import recommend_drugs

st.set_page_config(page_title="糖尿病治疗用药推荐", layout="wide")
st.title("糖尿病治疗用药推荐系统")
st.caption("基于《中国糖尿病防治指南（2024版）》及《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》")

# 加载数据库
@st.cache_data
def load_drug_db():
    return pd.read_csv("drugs_db.csv", encoding="utf-8-sig")

@st.cache_data
def load_fdc_db():
    try:
        df = pd.read_csv("fdc_db.csv", encoding="utf-8-sig")
        df["components_list"] = df["components"].apply(lambda x: [c.strip() for c in x.split(",")])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_recommendation_rules():
    return pd.read_csv("drug_recommendation.csv", encoding="utf-8-sig")

df_drugs = load_drug_db()
df_fdc = load_fdc_db()
df_rules = load_recommendation_rules()

FDC_CLINICAL_VALUE = """
FDC在指南中的临床价值（2025联合治疗共识，推荐意见17，IIa类推荐）：
1. 单药治疗血糖控制不达标者，可转换为FDC；
2. HbA1c较高需起始联合治疗的新诊断患者，可直接起始FDC；
3. 正在服用自由联合方案需简化治疗者，可转换为FDC。
FDC的优势：减少给药频率和漏服概率，提高依从性，是自由联合方案的合理替代选择。
"""

def find_matching_rule(rule_df, recommendation, patient_features):
    if recommendation.get("triple") and len(recommendation["triple"]) > 0:
        level = "triple"
    elif recommendation.get("dual") and len(recommendation["dual"]) > 0:
        level = "dual"
    else:
        level = "mono"

    if level == "triple":
        candidates = rule_df[rule_df["场景分类"].str.startswith("三联_")]
    elif level == "dual":
        candidates = rule_df[rule_df["场景分类"].str.startswith("联合_")]
    else:
        candidates = rule_df[rule_df["场景分类"].str.startswith("单药_")]

    if candidates.empty:
        return None, None, None, None

    for _, row in candidates.iterrows():
        condition = row["触发条件"]
        if evaluate_condition(condition, patient_features):
            return row["场景分类"], row["证据等级"], row["详细依据"], condition

    if not candidates.empty:
        row = candidates.iloc[0]
        return row["场景分类"], row["证据等级"], row["详细依据"], row["触发条件"]
    return None, None, None, None

def evaluate_condition(condition_str, features):
    if not condition_str or condition_str == "无":
        return False
    sub_conditions = condition_str.split(" 或 ")
    for cond in sub_conditions:
        cond = cond.strip()
        numbers = re.findall(r'(\d+\.?\d*)', cond)
        if "BMI ≥" in cond:
            if numbers:
                threshold = float(numbers[0])
                if features.get("bmi", 0) >= threshold:
                    return True
        elif "年龄≥" in cond:
            if numbers:
                threshold = float(numbers[0])
                if features.get("age", 0) >= threshold:
                    return True
        elif "合并ASCVD" in cond or "ASCVD" in cond:
            if features.get("has_ascvd", False):
                return True
        elif "合并HF" in cond or "心衰" in cond:
            if features.get("has_hf", False):
                return True
        elif "合并CKD" in cond or "CKD" in cond:
            if features.get("has_ckd", False):
                return True
        elif "低血糖风险高" in cond:
            if features.get("high_hypo_risk", False):
                return True
        elif "餐后血糖显著升高" in cond:
            if features.get("postprandial_high", False):
                return True
        elif "明显胰岛素抵抗" in cond:
            if features.get("insulin_resistance", False):
                return True
        elif "β细胞功能较好" in cond or "β细胞功能好" in cond:
            if features.get("beta_cell_good", False):
                return True
        elif "二甲双胍不耐受" in cond:
            if features.get("met_intolerance", False):
                return True
        elif "初诊HbA1c≥7.5%" in cond:
            if features.get("initial_high", False):
                return True
        elif "单药治疗3个月未达标" in cond:
            if features.get("single_failure", False):
                return True
        elif "二联治疗3个月未达标" in cond:
            if features.get("dual_failure", False):
                return True
        elif "二联未达标且胰岛素抵抗明显" in cond:
            if features.get("dual_failure", False) and features.get("insulin_resistance", False):
                return True
        elif "无特殊并发症且不胖" in cond:
            return True
    return False

def find_fdc_for_combo(combo_string):
    if df_fdc.empty:
        return []
    combo_lower = combo_string.lower()
    matched = []
    for _, row in df_fdc.iterrows():
        fdc_type = row.get("fdc_type", "").lower()
        if "sglt2" in combo_lower and "sglt2" in fdc_type:
            matched.append(row.to_dict())
        elif "dpp-4" in combo_lower and "dpp-4" in fdc_type:
            matched.append(row.to_dict())
        elif "tzd" in combo_lower and "tzd" in fdc_type:
            matched.append(row.to_dict())
        elif "su" in combo_lower and "su" in fdc_type:
            matched.append(row.to_dict())
    return matched

def parse_renal_rule(text, egfr):
    if pd.isna(text) or text == "" or text == "可用":
        return "可用"
    text = str(text)
    if "禁用" in text:
        patterns = [
            r"eGFR\s*<\s*(\d+)",
            r"<(\d+)",
            r"(\d+)\s*-\s*(\d+)\s*禁用",
            r"(\d+)\s*以下禁用"
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            for m in matches:
                if isinstance(m, tuple):
                    if len(m) == 1:
                        threshold = float(m[0])
                        if egfr < threshold:
                            return "禁用"
                    elif len(m) == 2:
                        low, high = float(m[0]), float(m[1])
                        if low <= egfr <= high:
                            return "禁用"
                else:
                    threshold = float(m)
                    if egfr < threshold:
                        return "禁用"
        if "eGFR<30" in text and egfr < 30:
            return "禁用"
        if "eGFR<25" in text and egfr < 25:
            return "禁用"
        if "eGFR<20" in text and egfr < 20:
            return "禁用"
        if "eGFR<15" in text and egfr < 15:
            return "禁用"
    if "减量慎用" in text or "减量" in text:
        patterns = [
            r"eGFR\s*(\d+)\s*-\s*(\d+)\s*减量",
            r"(\d+)\s*-\s*(\d+)\s*减量慎用",
            r"eGFR\s*<\s*(\d+)\s*减量",
            r"<(\d+)\s*减量"
        ]
        for pat in patterns:
            matches = re.findall(pat, text)
            for m in matches:
                if isinstance(m, tuple):
                    if len(m) == 2:
                        low, high = float(m[0]), float(m[1])
                        if low <= egfr <= high:
                            return "减量慎用"
                else:
                    threshold = float(m)
                    if egfr < threshold:
                        return "减量慎用"
        if "45-59" in text and 45 <= egfr <= 59:
            return "减量慎用"
        if "30-44" in text and 30 <= egfr <= 44:
            return "减量慎用"
        if "eGFR<45" in text and egfr < 45:
            return "减量慎用"
        if "eGFR<30" in text and egfr < 30:
            return "减量慎用"
        if "eGFR<25" in text and egfr < 25:
            return "减量慎用"
        if "eGFR<20" in text and egfr < 20:
            return "减量慎用"
        if "eGFR<15" in text and egfr < 15:
            return "减量慎用"
    if "慎用" in text and "减量" not in text:
        return "慎用"
    return "可用"

def parse_hepatic_rule(text, child_pugh):
    if pd.isna(text) or text == "" or text == "可用":
        return "可用"
    text = str(text)
    if "C禁用" in text and child_pugh == "C级（重度损伤）":
        return "禁用"
    if "B级" in text and child_pugh == "B级（中度损伤）":
        if "禁用" in text:
            return "禁用"
        elif "减量" in text:
            return "减量慎用"
        else:
            return "慎用"
    if "A级" in text and child_pugh == "A级（轻度损伤）":
        if "禁用" in text:
            return "禁用"
        elif "减量" in text:
            return "减量慎用"
        else:
            return "可用"
    if "Child-Pugh A/B" in text and child_pugh in ["A级（轻度损伤）", "B级（中度损伤）"]:
        return "可用"
    if "Child-Pugh C" in text and child_pugh == "C级（重度损伤）":
        return "禁用"
    if "ALT" in text or "AST" in text:
        return "需监测肝功能"
    return "可用"

# 患者信息输入
st.header("患者临床特征")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("年龄（岁）", 18, 100, 55)
    bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, 24.0)
    hba1c_current = st.number_input("当前 HbA1c (%)", 5.0, 15.0, 7.5, step=0.1)

with col2:
    st.subheader("合并症与高危因素")
    has_ascvd = st.checkbox("合并 ASCVD")
    has_hf = st.checkbox("合并 心衰 (HF)")
    has_ckd = st.checkbox("合并 慢性肾脏病 (CKD)")

with col3:
    st.subheader("临床特征")
    high_hypo_risk = st.checkbox("低血糖风险高")
    postprandial_high = st.checkbox("餐后血糖显著升高")
    insulin_resistance = st.checkbox("明显胰岛素抵抗")
    beta_cell_good = st.checkbox("β细胞功能较好")
    met_intolerance = st.checkbox("二甲双胍不耐受")

st.subheader("治疗阶段")
treatment_stage = st.radio(
    "当前阶段",
    ["未用药/初始治疗", "单药治疗 ≥3个月", "二联治疗 ≥3个月"],
    horizontal=True
)

hba1c_after_single = None
hba1c_after_dual = None
if treatment_stage == "单药治疗 ≥3个月":
    hba1c_after_single = st.number_input("单药后 HbA1c (%)", 5.0, 15.0, 7.5, step=0.1)
elif treatment_stage == "二联治疗 ≥3个月":
    hba1c_after_dual = st.number_input("二联后 HbA1c (%)", 5.0, 15.0, 7.8, step=0.1)

# 生成个性化药物推荐
def generate_recommendation():
    with st.spinner("正在根据指南分析..."):
        recommendation = recommend_drugs(
            age=age,
            bmi=bmi,
            hba1c_current=hba1c_current,
            has_ascvd=has_ascvd,
            has_hf=has_hf,
            has_ckd=has_ckd,
            high_hypo_risk=high_hypo_risk,
            postprandial_high=postprandial_high,
            insulin_resistance=insulin_resistance,
            beta_cell_good=beta_cell_good,
            met_intolerance=met_intolerance,
            treatment_stage=treatment_stage,
            hba1c_after_single=hba1c_after_single,
            hba1c_after_dual=hba1c_after_dual
        )
        # 保存到 session_state
        st.session_state['recommendation'] = recommendation
        # 同时保存患者特征，以便规则匹配
        patient_features = {
            "age": age,
            "bmi": bmi,
            "hba1c_current": hba1c_current,
            "has_ascvd": has_ascvd,
            "has_hf": has_hf,
            "has_ckd": has_ckd,
            "high_hypo_risk": high_hypo_risk,
            "postprandial_high": postprandial_high,
            "insulin_resistance": insulin_resistance,
            "beta_cell_good": beta_cell_good,
            "met_intolerance": met_intolerance,
            "treatment_stage": treatment_stage,
            "single_failure": recommendation["single_failure"],
            "dual_failure": recommendation["dual_failure"],
            "initial_high": (treatment_stage == "未用药/初始治疗" and hba1c_current >= 7.5)
        }
        st.session_state['patient_features'] = patient_features
        # 同时保存单药/二联后的HbA1c，以便用于规则匹配（可选）
        st.session_state['hba1c_after_single'] = hba1c_after_single
        st.session_state['hba1c_after_dual'] = hba1c_after_dual

run_recommend = st.button("生成推荐方案", type="primary", use_container_width=True, on_click=generate_recommendation)

# ==================== 主页面 - 使用 st.radio 模拟 Tab ====================
# 添加自定义 CSS 使 radio 伪装成 Tab 样式
st.markdown("""
<style>
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        gap: 0px;
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 4px;
    }
    div[data-testid="stRadio"] > div label {
        flex: 1;
        text-align: center;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s;
        color: #333;
        font-weight: 500;
        margin: 0;
        background-color: transparent;
    }
    div[data-testid="stRadio"] > div label:hover {
        background-color: #e0e4ea;
    }
    /* 隐藏原始的 radio 圆点 */
    div[data-testid="stRadio"] > div label > div:first-child {
        display: none;
    }
    /* 选中状态 */
    div[data-testid="stRadio"] > div label[data-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

tab_names = [
    "药物信息浏览",
    "药物对比分析",
    "个体化用药推荐",
    "肝肾功能剂量调整",
    "药物相互作用检查"
]
selected_tab = st.radio(
    "导航",
    tab_names,
    index=0,
    horizontal=True,
    key="main_tab_radio",
    label_visibility="collapsed"
)

# 根据选择显示对应内容
if selected_tab == "药物信息浏览":
    with st.container():
        # 原 tab1 的内容
        st.subheader("药物信息浏览")
        drug_classes = df_drugs["drug_class"].unique().tolist()
        selected_class = st.selectbox("按药物分类筛选", ["全部"] + drug_classes)
        display_df = df_drugs if selected_class == "全部" else df_drugs[df_drugs["drug_class"] == selected_class]
        st.write(f"共 {len(display_df)} 种药物")

        cols = st.columns(3)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            col = cols[idx % 3]
            with col:
                with st.expander(f"{row['drug_name_cn']} ({row['drug_class']})"):
                    st.markdown(f"**英文名**：{row['drug_name_en']}")
                    st.markdown(f"**作用机制**：{row['mechanism']}")
                    st.markdown(f"**靶点**：{row['target']}")
                    st.markdown(f"**降糖效力**：{row['hba1c_reduction']}")
                    st.markdown(f"**低血糖风险**：{row['hypoglycemia_risk']}")
                    st.markdown(f"**体重影响**：{row['weight_effect']}")
                    st.markdown(f"**心肾获益**：{row['cv_benefit']}")
                    st.markdown(f"**禁忌症**：{row['contraindications']}")
                    if pd.notna(row['cid']):
                        try:
                            st.image(
                                f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={int(row['cid'])}&t=l",
                                caption="分子结构", width=150
                            )
                        except:
                            pass

        st.markdown("---")
        with st.expander("查看所有固定剂量复方制剂（FDC）"):
            st.markdown(FDC_CLINICAL_VALUE)
            st.markdown("---")
            if not df_fdc.empty:
                st.dataframe(
                    df_fdc[["name", "components", "specification", "drug_class", "advantages", "caution"]],
                    use_container_width=True,
                    hide_index=True
                )
            st.caption("当推荐方案匹配到FDC时，会在「个体化用药推荐」结果中自动提示简化方案")

elif selected_tab == "药物对比分析":
    with st.container():
        # 原 tab2 的内容
        st.subheader("药物对比分析")
        all_drug_names = df_drugs["drug_name_cn"].tolist()
        selected_drugs = st.multiselect("选择2-4种药物进行对比", all_drug_names, default=all_drug_names[:3])

        if len(selected_drugs) >= 2:
            compare_df = df_drugs[df_drugs["drug_name_cn"].isin(selected_drugs)]

            st.write("多维对比表")
            display_cols = ["drug_name_cn", "drug_class", "mechanism", "target", "hba1c_reduction",
                            "hypoglycemia_risk", "weight_effect", "cv_benefit", "contraindications"]
            st.dataframe(compare_df[display_cols], use_container_width=True, hide_index=True)

            st.write("能力雷达图")
            score_map = {
                "低": 3, "较低": 2.5, "中等": 2, "较高": 1, "高": 0,
                "中性或轻度减轻": 2, "轻度减轻": 2, "减轻": 2.5, "显著减轻": 3,
                "中性": 1.5, "增加": 0,
                "获益": 3, "潜在获益": 2, "证据不足": 0,
            }
            fig = go.Figure()
            dimensions = ["降糖效力", "低血糖安全", "体重获益", "心肾获益"]
            for _, row in compare_df.iterrows():
                scores = []
                hba1c = row["hba1c_reduction"]
                if "%~" in str(hba1c):
                    nums = [float(x) for x in str(hba1c).replace("%", "").split("~")]
                    scores.append((nums[0] + nums[1]) / 2)
                else:
                    scores.append(0.5)
                scores.append(score_map.get(row["hypoglycemia_risk"], 1))
                scores.append(score_map.get(row["weight_effect"], 1))
                scores.append(score_map.get(row["cv_benefit"], 1))
                fig.add_trace(go.Scatterpolar(r=scores, theta=dimensions, fill='toself', name=row["drug_name_cn"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3.5])), height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("请至少选择2种药物进行对比")

elif selected_tab == "个体化用药推荐":
    with st.container():
        # 原 tab3 的内容
        st.subheader("个体化用药推荐")
        st.caption("基于《中国糖尿病防治指南（2024版）》及《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》")

        if 'recommendation' in st.session_state:
            recommendation = st.session_state['recommendation']
            patient_features = st.session_state.get('patient_features', {})
            if not patient_features:
                patient_features = {
                    "age": age,
                    "bmi": bmi,
                    "hba1c_current": hba1c_current,
                    "has_ascvd": has_ascvd,
                    "has_hf": has_hf,
                    "has_ckd": has_ckd,
                    "high_hypo_risk": high_hypo_risk,
                    "postprandial_high": postprandial_high,
                    "insulin_resistance": insulin_resistance,
                    "beta_cell_good": beta_cell_good,
                    "met_intolerance": met_intolerance,
                    "treatment_stage": treatment_stage,
                    "single_failure": recommendation["single_failure"],
                    "dual_failure": recommendation["dual_failure"],
                    "initial_high": (treatment_stage == "未用药/初始治疗" and hba1c_current >= 7.5)
                }
                st.session_state['patient_features'] = patient_features

            scenario, evidence_level, evidence_detail, trigger = find_matching_rule(
                df_rules, recommendation, patient_features
            )

            if recommendation["single_failure"]:
                st.warning("单药治疗未达标（HbA1c ≥ 7.0%），建议启动二联治疗")
            if recommendation["dual_failure"]:
                st.warning("二联治疗未达标（HbA1c ≥ 7.0%），建议启动三联治疗")

            st.markdown("### 决策依据")
            if scenario:
                st.markdown(
                    f"""
                    <div style="border:2px solid #4CAF50; border-radius:8px; padding:16px; background-color:#f0fff4; margin-bottom:16px;">
                        <p style="font-size:16px; font-weight:bold; color:#2E7D32;">匹配场景：{scenario}</p>
                        <p><strong>触发条件：</strong>{trigger}</p>
                        <p><strong>证据等级：</strong><span style="background-color:#FFD700; padding:2px 8px; border-radius:4px; font-weight:bold;">{evidence_level}</span></p>
                        <p><strong>详细依据：</strong>{evidence_detail}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("未能从规则库中精确匹配场景，以下推荐基于指南常规路径。")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**单药推荐**")
                if recommendation["mono"]:
                    for drug in recommendation["mono"]:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #4CAF50; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#E8F5E9;">
                                <span style="font-size:16px; font-weight:bold; color:#2E7D32;">{drug}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("暂不需要单药治疗")

            with col2:
                st.markdown("**二联推荐**")
                if recommendation["dual"]:
                    for combo in recommendation["dual"]:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #2196F3; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#E3F2FD;">
                                <span style="font-size:16px; font-weight:bold; color:#0D47A1;">{combo}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        fdc_matches = find_fdc_for_combo(combo)
                        if fdc_matches:
                            for fdc in fdc_matches:
                                st.markdown(
                                    f"""
                                    <div style="border:1px dashed #FF9800; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#FFF3E0; margin-left:12px;">
                                        <span style="font-weight:bold;">简化方案（FDC）：</span>{fdc['name']}
                                        <br><span style="font-size:13px; color:#555;">成分：{fdc['components']} | 规格：{fdc['specification']}</span>
                                        <br><span style="font-size:13px; color:#555;">优势：{fdc['advantages']} | 注意：{fdc['caution']}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            st.caption("FDC将两种药物制成单片，每日1次，可显著提高依从性（指南IIa类推荐）")
                else:
                    st.info("暂不需要二联治疗")

            with col3:
                st.markdown("**三联推荐**")
                if recommendation["triple"]:
                    for combo in recommendation["triple"]:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #F44336; border-radius:6px; padding:10px; margin-bottom:8px; background-color:#FFEBEE;">
                                <span style="font-size:16px; font-weight:bold; color:#B71C1C;">{combo}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("暂不需要三联治疗")

            # 推荐药物详细指导
            st.markdown("---")
            st.subheader("推荐药物详细指导")
            
            category_keywords = {"DPP-4i", "SGLT2i", "SU", "GLN", "AGi", "TZD", "GKA", "GLP-1RA", "pan-PPARA", "双胍类"}

            all_recommended_raw = recommendation["mono"] + recommendation["dual"] + recommendation["triple"]
            valid_drugs_set = set()
            invalid_categories_set = set()

            for item in all_recommended_raw:
                # 按常见分隔符拆分
                parts = re.split(r'[+、，, 或]', item)
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue

                    if p in df_drugs["drug_name_cn"].values:
                        valid_drugs_set.add(p)
                        continue

                    match = re.search(r'（([^）]+)）', p)
                    if match:
                        inner = match.group(1).strip()
                        if inner in df_drugs["drug_name_cn"].values:
                            valid_drugs_set.add(inner)
                            continue
                        found = False
                        for drug in df_drugs["drug_name_cn"].values:
                            if inner in drug or drug in inner:
                                valid_drugs_set.add(drug)
                                found = True
                                break
                        if found:
                            continue

                    clean_p = re.sub(r'[（()）]', '', p).strip()
                    if clean_p in category_keywords:
                        invalid_categories_set.add(clean_p)
                        continue

                    found = False
                    for drug in df_drugs["drug_name_cn"].values:
                        if clean_p in drug or drug in clean_p:
                            valid_drugs_set.add(drug)
                            found = True
                            break
                    if found:
                        continue

            valid_drugs = sorted(list(valid_drugs_set))

            if invalid_categories_set:
                st.info(f"以下推荐为纯药物类别（{', '.join(sorted(invalid_categories_set))}），无法查看单一药物详情，请参照具体药物名称。")

            if valid_drugs:
                guide_drug = st.selectbox(
                    "选择推荐中的具体药物查看详细指导卡片",
                    ["请选择"] + valid_drugs,
                    key="guide_in_tab3"
                )
                if guide_drug and guide_drug != "请选择":
                    matched = df_drugs[df_drugs["drug_name_cn"] == guide_drug]  # 精确匹配
                    if not matched.empty:
                        row = matched.iloc[0]
                        st.markdown(
                            f"""
                            <div style="border:2px solid #2196F3; border-radius:8px; padding:16px; background-color:#F5F9FF; margin-top:8px;">
                                <h4 style="color:#0D47A1;">{row['drug_name_cn']} ({row['drug_name_en']})</h4>
                                <table style="width:100%; border-collapse:collapse; font-size:14px;">
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">药物分类</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['drug_class']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">作用机制</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['mechanism']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">靶点</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['target']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">降糖效力</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hba1c_reduction']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">低血糖风险</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hypoglycemia_risk']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">体重影响</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['weight_effect']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">心肾获益</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['cv_benefit']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">禁忌症</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['contraindications']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">肾功能调整</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['renal_dose_adjustment']}</td></tr>
                                    <tr><td style="padding:6px 8px; font-weight:bold; border-bottom:1px solid #ddd;">肝功能使用</td><td style="padding:6px 8px; border-bottom:1px solid #ddd;">{row['hepatic_use']}</td></tr>
                                </table>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if pd.notna(row['cid']):
                            try:
                                st.image(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={int(row['cid'])}&t=l", caption="分子结构", width=150)
                            except:
                                pass
                    else:
                        st.info(f"未找到 {guide_drug} 的详细信息")
            else:
                st.info("暂无具体的推荐药物可查看详情")

        else:
            st.info("请填写患者特征后，点击「生成推荐方案」")

elif selected_tab == "肝肾功能剂量调整":
    with st.container():
        # 原 tab4 的内容
        st.subheader("肝肾功能剂量调整")
        st.markdown(
            """
            <div style="border:1px solid #FF9800; border-radius:6px; padding:12px; background-color:#FFF8E1; margin-bottom:16px;">
                <strong>功能说明：</strong>输入患者的肾功能（eGFR）和肝功能分级，系统将自动根据CSV中的剂量调整建议，判断每种降糖药是否需要调整剂量或禁用。
                数据来源于《成人2型糖尿病口服降糖药联合治疗专家共识（2025版）》,  eGFR 数值对应分级基于 KDIGO 指南的 CKD 分期推荐的代表性值。
            </div>
            """,
            unsafe_allow_html=True
        )

        # 肾功能
        def parse_renal_rule(text, egfr):
            if pd.isna(text) or text == "" or text == "可用":
                return "可用"
            text = str(text)

            if "禁用" in text:
                patterns = [
                    r"eGFR\s*<\s*(\d+)",
                    r"<(\d+)",
                    r"(\d+)\s*-\s*(\d+)\s*禁用",
                    r"(\d+)\s*以下禁用"
                ]
                for pat in patterns:
                    matches = re.findall(pat, text)
                    for m in matches:
                        if isinstance(m, tuple):
                            if len(m) == 1:
                                if egfr < float(m[0]):
                                    return "禁用"
                            elif len(m) == 2:
                                low, high = float(m[0]), float(m[1])
                                if low <= egfr <= high:
                                    return "禁用"
                        else:
                            if egfr < float(m):
                                return "禁用"

            if "减量慎用" in text or "减量" in text:
                patterns = [
                    r"eGFR\s*(\d+)\s*-\s*(\d+)\s*减量",
                    r"(\d+)\s*-\s*(\d+)\s*减量慎用",
                    r"eGFR\s*<\s*(\d+)\s*减量",
                    r"<(\d+)\s*减量"
                ]
                for pat in patterns:
                    matches = re.findall(pat, text)
                    for m in matches:
                        if isinstance(m, tuple):
                            if len(m) == 2:
                                low, high = float(m[0]), float(m[1])
                                if low <= egfr <= high:
                                    return "减量慎用"
                        else:
                            if egfr < float(m):
                                return "减量慎用"
       
            if "慎用" in text and "减量" not in text:
                patterns = [
                    r"eGFR\s*<\s*(\d+)",
                    r"<(\d+)",
                    r"eGFR\s*(\d+)\s*-\s*(\d+)\s*慎用",
                    r"(\d+)\s*-\s*(\d+)\s*慎用"
                ]
                for pat in patterns:
                    matches = re.findall(pat, text)
                    for m in matches:
                        if isinstance(m, tuple):
                            if len(m) == 1:
                                if egfr < float(m[0]):
                                    return "慎用"
                            elif len(m) == 2:
                                low, high = float(m[0]), float(m[1])
                                if low <= egfr <= high:
                                    return "慎用"
                        else:
                            if egfr < float(m):
                                return "慎用"

            return "可用"   # 默认可用

        # 肝功能
        def parse_hepatic_rule(text, child_pugh):
            if pd.isna(text) or text == "" or text == "可用":
                return "可用"
            text = str(text)

            if child_pugh == "不详/未评估":
                return "可用"

            if "中度以上禁用" in text:
                if child_pugh in ["B级（中度损伤）", "C级（重度损伤）"]:
                    return "禁用"
                else:
                    return "可用"

            if "Child-Pugh A/B可用" in text and child_pugh in ["A级（轻度损伤）", "B级（中度损伤）"]:
                return "可用"

            if "Child-Pugh C" in text and child_pugh == "C级（重度损伤）":
                return "禁用"

            if "可用（A/B/C级" in text:
                return "可用"

            # ALT/AST 监测
            if "ALT" in text or "AST" in text:
                return "需监测肝功能"

            if "禁用" in text:
                return "禁用"
            if "减量" in text or "减量慎用" in text:
                return "减量慎用"
            if "慎用" in text:
                return "慎用"

            return "可用"

        col1, col2 = st.columns(2)
        with col1:
            egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0, max_value=120, value=60, step=5)
            st.caption("参考范围：≥60正常，45-59轻中度下降，30-44中重度下降，15-29重度下降，<15肾衰竭")
        with col2:
            child_pugh = st.selectbox(
                "肝功能分级 (Child-Pugh)",
                ["A级（轻度损伤）", "B级（中度损伤）", "C级（重度损伤）", "不详/未评估"]
            )

        if st.button("评估所有药物的剂量调整建议", type="primary"):
            st.markdown("### 剂量调整评估结果")
            st.caption("根据CSV中的`renal_dose_adjustment`和`hepatic_use`字段进行匹配，颜色标识：绿色=可用，黄色=减量慎用，红色=禁用")

            results = []
            for _, row in df_drugs.iterrows():
                drug_name = row["drug_name_cn"]
                renal_text = row["renal_dose_adjustment"]
                hepatic_text = row["hepatic_use"]
                renal_advice = parse_renal_rule(renal_text, egfr)
                hepatic_advice = parse_hepatic_rule(hepatic_text, child_pugh)

                if "禁用" in renal_advice or "禁用" in hepatic_advice:
                    status = "禁用"
                    color = "#FFEBEE"
                    border = "#F44336"
                elif "需监测肝功能" in hepatic_advice:
                    status = "减量慎用"
                    color = "#FFF8E1"
                    border = "#FF9800"
                elif "减量慎用" in renal_advice or "减量慎用" in hepatic_advice or "慎用" in renal_advice or "慎用" in hepatic_advice:
                    status = "减量慎用"
                    color = "#FFF8E1"
                    border = "#FF9800"
                else:
                    status = "可用"
                    color = "#E8F5E9"
                    border = "#4CAF50"

                results.append({
                    "药物": drug_name,
                    "分类": row["drug_class"],
                    "eGFR建议": renal_advice,
                    "肝功能建议": hepatic_advice,
                    "综合建议": status,
                    "color": color,
                    "border": border
                })

            status_order = {"禁用": 0, "减量慎用": 1, "可用": 2}
            results_sorted = sorted(results, key=lambda x: status_order[x["综合建议"]])

            for r in results_sorted:
                st.markdown(
                    f"""
                    <div style="border-left:6px solid {r['border']}; border-radius:4px; padding:10px 14px; margin-bottom:8px; background-color:{r['color']};">
                        <span style="font-size:16px; font-weight:bold;">{r['药物']}</span>
                        <span style="font-size:13px; color:#555; margin-left:12px;">{r['分类']}</span>
                        <span style="float:right; font-weight:bold; color:{r['border']};">
                            {r['综合建议']}
                        </span>
                        <br><span style="font-size:13px; color:#555;">eGFR建议：{r['eGFR建议']} ｜ 肝功能建议：{r['肝功能建议']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption("提示：具体建议根据CSV字段解析，如有疑问请参考药品说明书。")

elif selected_tab == "药物相互作用检查":
    with st.container():
        # 原 tab5 的内容
        st.subheader("药物相互作用检查")
        st.caption("选择患者当前正在使用的所有药物，系统将检查已知的相互作用")

        all_drugs_for_interaction = df_drugs["drug_name_cn"].tolist() + [
            "ACEI类降压药", "ARB类降压药", "他汀类降脂药", "贝特类降脂药",
            "阿司匹林", "氯吡格雷", "利尿剂", "β受体阻滞剂",
            "胰岛素", "碘化对比剂（造影检查）", "糖皮质激素"
        ]

        current_drugs = st.multiselect("选择患者正在使用的药物", all_drugs_for_interaction)

        if current_drugs:
            st.write(f"**当前用药**：{'、'.join(current_drugs)}")

            interactions = []

            if any("二甲双胍" in d for d in current_drugs) and any("碘化对比剂" in d for d in current_drugs):
                interactions.append({
                    "level": "谨慎",
                    "color": "#FF9800",
                    "bg": "#FFF8E1",
                    "drugs": "二甲双胍 + 碘化对比剂",
                    "advice": "检查当日停用二甲双胍，检查完至少48h后复查肾功能无变化方可继续使用",
                    "evidence": "二甲双胍临床应用专家共识"
                })

            sglt2_drugs = ["恩格列净", "达格列净", "卡格列净", "艾托格列净", "恒格列净", "加格列净"]
            if any(d in sglt2_drugs for d in current_drugs):
                if any("糖皮质激素" in d for d in current_drugs):
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "SGLT2i + 糖皮质激素",
                        "advice": "糖皮质激素可升高血糖，需加强血糖监测，可能需要增加降糖药剂量",
                        "evidence": "OAD联合治疗注意事项"
                    })

            if any("沙格列汀" in d for d in current_drugs) and has_hf:
                interactions.append({
                    "level": "谨慎",
                    "color": "#FF9800",
                    "bg": "#FFF8E1",
                    "drugs": "沙格列汀 + 心衰",
                    "advice": "SAVOR研究显示沙格列汀可能增加心衰住院风险，有心衰诱发因素的患者慎用",
                    "evidence": "DPP-4i CVOT研究结果"
                })

            su_drugs = ["格列美脲", "格列吡嗪", "格列齐特", "格列喹酮"]
            if any(d in su_drugs for d in current_drugs):
                if age >= 65:
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "SU类 + 高龄(≥65岁)",
                        "advice": "老年患者低血糖感知能力下降，建议减量或换用低血糖风险更小的药物",
                        "evidence": "老年T2DM用药注意事项"
                    })
                if high_hypo_risk:
                    interactions.append({
                        "level": "高警示",
                        "color": "#F44336",
                        "bg": "#FFEBEE",
                        "drugs": "SU类 + 低血糖高危",
                        "advice": "患者有低血糖高危因素，使用SU类药物风险显著增加，建议换用DPP-4i或SGLT2i",
                        "evidence": "OAD联合治疗注意事项"
                    })

            tzd_drugs = ["吡格列酮", "罗格列酮"]
            if any(d in tzd_drugs for d in current_drugs):
                if has_hf:
                    interactions.append({
                        "level": "禁忌",
                        "color": "#F44336",
                        "bg": "#FFEBEE",
                        "drugs": "TZD + 心衰",
                        "advice": "TZD类药物有水钠潴留风险，NYHA II级以上心衰患者禁用",
                        "evidence": "TZD使用注意事项"
                    })
                if age >= 65:
                    interactions.append({
                        "level": "谨慎",
                        "color": "#FF9800",
                        "bg": "#FFF8E1",
                        "drugs": "TZD + 高龄(≥65岁)",
                        "advice": "老年患者骨质疏松和骨折风险增加，TZD慎用",
                        "evidence": "老年T2DM用药注意事项"
                    })

            if interactions:
                st.subheader("相互作用检查结果")
                for inter in interactions:
                    st.markdown(
                        f"""
                        <div style="border-left:6px solid {inter['color']}; border-radius:4px; padding:12px 16px; margin-bottom:12px; background-color:{inter['bg']};">
                            <span style="font-weight:bold; font-size:16px; color:{inter['color']};">{inter['level']}</span>
                            <span style="font-weight:bold; margin-left:12px;">{inter['drugs']}</span>
                            <br><span style="color:#333;">{inter['advice']}</span>
                            <br><span style="font-size:13px; color:#777;">依据：{inter['evidence']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.success("未发现已知的药物相互作用，当前用药方案安全。")

            st.info("通用建议：任何用药方案调整后，均应加强血糖监测，特别是联合使用胰岛素促泌剂（SU/格列奈类）或胰岛素时。")
        else:
            st.info("请选择患者正在使用的药物进行相互作用检查")
