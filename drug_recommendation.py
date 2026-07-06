def recommend_drugs(
    # 基础特征
    age,
    bmi,
    hba1c_current,
    # 合并症
    has_ascvd=False,
    has_hf=False,
    has_ckd=False,
    # 临床特征
    high_hypo_risk=False,
    postprandial_high=False,
    insulin_resistance=False,
    beta_cell_good=False,
    met_intolerance=False,
    # 治疗阶段与达标情况
    treatment_stage="未用药/初始治疗",  # "未用药/初始治疗", "单药治疗 ≥3个月", "二联治疗 ≥3个月"
    hba1c_after_single=None,  # 如果单药治疗，提供治疗后HbA1c
    hba1c_after_dual=None,    # 如果二联治疗，提供治疗后HbA1c
):
    # 判断是否未达标
    single_failure = False
    dual_failure = False

    if treatment_stage == "单药治疗 ≥3个月":
        if hba1c_after_single is not None and hba1c_after_single >= 7.0:
            single_failure = True
    elif treatment_stage == "二联治疗 ≥3个月":
        if hba1c_after_dual is not None and hba1c_after_dual >= 7.0:
            dual_failure = True

    # 初始治疗时，若HbA1c≥7.5%，直接联合用药
    initial_high = (treatment_stage == "未用药/初始治疗" and hba1c_current >= 7.5)
    # 初始治疗时，若HbA1c≥9.0%，直接考虑三联治疗或胰岛素强化
    severe_hyperglycemia = (treatment_stage == "未用药/初始治疗" and hba1c_current >= 9.0)
    # 判断是否需要二联
    need_dual = single_failure or initial_high
    # 判断是否需要三联
    need_triple = dual_failure or severe_hyperglycemia

    # 构建推荐方案
    result = {
        "mono": [],
        "dual": [],
        "triple": [],
        "status": treatment_stage,
        "single_failure": single_failure,
        "dual_failure": dual_failure,
    }

    # 单药推荐
    if not need_dual and not need_triple:
        if has_ascvd or has_hf or has_ckd:
            result["mono"].append("SGLT2i 或 GLP-1RA")
        elif bmi >= 24:
            result["mono"].append("GLP-1RA（司美格鲁肽）或 SGLT2i")
        elif high_hypo_risk or age >= 65:
            result["mono"].append("DPP-4i 或 二甲双胍")
        elif postprandial_high:
            result["mono"].append("α-糖苷酶抑制剂（阿卡波糖）")
        elif insulin_resistance:
            result["mono"].append("TZD（吡格列酮）或 PPAR泛激动剂（西格列他钠）")
        elif beta_cell_good:
            result["mono"].append("GKA（多格列艾汀）")
        else:
            result["mono"].append("二甲双胍")
        
        # 二甲双胍不耐受
        if met_intolerance and "二甲双胍" in result["mono"]:
            result["mono"] = ["DPP-4i（西格列汀）", "SGLT2i（达格列净）"]
    
    # 二联推荐
    if need_dual and not need_triple:
        base = "" if met_intolerance else "Met + "
        if has_ascvd or has_hf or has_ckd:
            result["dual"].append("Met + SGLT2i" if not met_intolerance else "SGLT2i + GLP-1RA")
            if not met_intolerance:
                result["dual"].append("Met + GLP-1RA")
        elif bmi >= 24:
            result["dual"].append("Met + GLP-1RA" if not met_intolerance else "GLP-1RA + SGLT2i")
            if not met_intolerance:
                result["dual"].append("Met + SGLT2i")
        elif high_hypo_risk or age >= 65:
            result["dual"].append("Met + DPP-4i" if not met_intolerance else "DPP-4i + SGLT2i")
        elif postprandial_high:
            result["dual"].append("Met + α-糖苷酶抑制剂（阿卡波糖）")
        elif insulin_resistance:
            result["dual"].append("Met + TZD（吡格列酮）")
            if not met_intolerance:
                result["dual"].append("Met + PPAR泛激动剂（西格列他钠）")
        elif beta_cell_good:
            result["dual"].append("Met + GKA（多格列艾汀）")
            if not met_intolerance:
                result["dual"].append("Met + DPP-4i")
        else:
            # 默认二联
            result["dual"].append("Met + DPP-4i" if not met_intolerance else "DPP-4i + SGLT2i")
        
        # Met 不耐受
        if met_intolerance:
            result["dual"] = [opt.replace("Met + ", "") for opt in result["dual"] if "Met" in opt] or ["DPP-4i + SGLT2i"]
            if "Met" not in result["dual"][0]:
                result["dual"] = ["DPP-4i + SGLT2i", "SGLT2i + GLP-1RA"]

    # 三联推荐
    if need_triple:
        if has_ascvd or has_ckd or has_hf or bmi >= 24:
            result["triple"].append("Met + SGLT2i + GLP-1RA" if not met_intolerance else "SGLT2i + GLP-1RA + DPP-4i")
            if not met_intolerance:
                result["triple"].append("Met + DPP-4i + SGLT2i")
        elif insulin_resistance:
            result["triple"].append("Met + DPP-4i + TZD" if not met_intolerance else "DPP-4i + SGLT2i + TZD")
        else:
            result["triple"].append("Met + DPP-4i + SGLT2i" if not met_intolerance else "SGLT2i + GLP-1RA + DPP-4i")

    return result