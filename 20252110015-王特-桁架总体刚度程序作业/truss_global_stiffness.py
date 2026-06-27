import numpy as np
import json

# ====================== 1. 前处理工具函数 ======================
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_LM(IEN, nnp, nel, ndof):
    """通用LM矩阵生成，兼容ndof=1(一维)、ndof=2(二维)"""
    nen = 2
    LM = np.zeros((nen * ndof, nel), dtype=int)
    for e in range(nel):
        n1 = IEN[e][0] - 1
        n2 = IEN[e][1] - 1
        # 节点i全部自由度
        for d in range(ndof):
            LM[d, e] = n1 * ndof + d
        # 节点j全部自由度
        for d in range(ndof):
            LM[ndof + d, e] = n2 * ndof + d
    return LM

# ====================== 2. 单元刚度计算 ======================
def element_2d_truss(node_i, node_j, E, A):
    xi, yi = node_i
    xj, yj = node_j
    dx = xj - xi
    dy = yj - yi
    L = np.sqrt(dx**2 + dy**2)
    c = dx / L
    s = dy / L
    ke = (E * A / L) * np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])
    return ke, L, c, s

def element_1d_bar(xi, xj, EA_L):
    L = abs(xj - xi)
    ke = EA_L * np.array([[1, -1], [-1, 1]])
    return ke, L

# ====================== 3. 总体刚度组装 ======================
def assemble_global(K, Ke, LM, e):
    ndof_elem = Ke.shape[0]
    for a in range(ndof_elem):
        for b in range(ndof_elem):
            gdof_a = LM[a, e]
            gdof_b = LM[b, e]
            K[gdof_a, gdof_b] += Ke[a, b]
    return K

# ====================== 4. 缩减法边界条件求解 ======================
def solve_reduction(K, f, fixed_dof, fixed_val):
    ndof_total = K.shape[0]
    all_dof = np.arange(ndof_total)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    nF = len(free_dof)
    nE = len(fixed_dof)
    # 分块矩阵
    KFF = K[np.ix_(free_dof, free_dof)]
    KFE = K[np.ix_(free_dof, fixed_dof)]
    KEF = K[np.ix_(fixed_dof, free_dof)]
    KEE = K[np.ix_(fixed_dof, fixed_dof)]
    fF = f[free_dof]
    dE = np.array(fixed_val)
    # 求解未知位移
    dF = np.linalg.solve(KFF, fF - KFE @ dE)
    # 重构完整位移
    d = np.zeros(ndof_total)
    d[fixed_dof] = dE
    d[free_dof] = dF
    # 计算支座反力
    fE = KEE @ dE + KEF @ dF
    return d, fE, free_dof

# ====================== 5. 后处理：单元应力、轴力 ======================
def calc_stress_2d(d_elem, E, L, c, s):
    sigma = E / L * (-c * d_elem[0] - s * d_elem[1] + c * d_elem[2] + s * d_elem[3])
    N = sigma * A
    return sigma, N

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 切换算例：1=一维两杆，2=二维桁架
    case_id = 1
    if case_id == 1:
        # ========== 算例1：一维两单元杆 ==========
        print("========== 算例1：一维两单元杆结构 ==========")
        nnp = 3
        ndof = 1
        nel = 2
        IEN = [[1, 2], [2, 3]]
        x = [0, 1, 2]
        EA_L_list = [100, 200]
        fixed_dof = [0]
        fixed_val = [0.0]
        force_dof = [2]
        force_val = [10.0]
        ndof_total = nnp * ndof
        K = np.zeros((ndof_total, ndof_total))
        LM = build_LM(IEN, nnp, nel, ndof)
        # 组装
        for e in range(nel):
            n1 = IEN[e][0]-1
            n2 = IEN[e][1]-1
            xi, xj = x[n1], x[n2]
            Ke, L = element_1d_bar(xi, xj, EA_L_list[e])
            K = assemble_global(K, Ke, LM, e)
        print("组装后总体刚度矩阵K：")
        print(K)
        # 校验对称
        sym_err = np.linalg.norm(K - K.T)
        print(f"刚度矩阵对称误差：{sym_err:.2e}")
        # 载荷向量
        f = np.zeros(ndof_total)
        for idx, dof in enumerate(force_dof):
            f[dof] = force_val[idx]
        # 求解
        d, f_react, free = solve_reduction(K, f, fixed_dof, fixed_val)
        print("\n全局节点位移d：", d)
        print("支座约束反力：", f_react)

    elif case_id == 2:
        # ========== 算例2：二维两杆桁架 ==========
        print("========== 算例2：二维两杆桁架结构 ==========")
        data = load_json("case2_2d.json")
        nsd = data["nsd"]
        ndof = data["ndof"]
        nnp = data["nnp"]
        nel = data["nel"]
        E_list = data["E"]
        A_list = data["CArea"]
        x = data["x"]
        y = data["y"]
        IEN = data["IEN"]
        fixed_dof = np.array(data["fixed_dof"]) - 1
        fixed_val = data["fixed_value"]
        force_dof = np.array(data["force_dof"]) - 1
        force_val = data["force_value"]
        ndof_total = nnp * ndof
        K = np.zeros((ndof_total, ndof_total))
        LM = build_LM(IEN, nnp, nel, ndof)
        print("LM对号矩阵：")
        print(LM)
        elem_info = []
        # 单元循环组装
        for e in range(nel):
            n1 = IEN[e][0]-1
            n2 = IEN[e][1]-1
            node_i = [x[n1], y[n1]]
            node_j = [x[n2], y[n2]]
            E = E_list[e]
            A = A_list[e]
            Ke, L, c, s = element_2d_truss(node_i, node_j, E, A)
            K = assemble_global(K, Ke, LM, e)
            elem_info.append([L, c, s, E, A])
        print("\n总体刚度矩阵K：")
        print(np.round(K,4))
        sym_err = np.linalg.norm(K - K.T)
        print(f"刚度对称误差：{sym_err:.2e}")
        # 载荷向量
        f = np.zeros(ndof_total)
        for idx, dof in enumerate(force_dof):
            f[dof] = force_val[idx]
        # 求解位移、反力
        d, f_react, free = solve_reduction(K, f, fixed_dof, fixed_val)
        print("\n全局节点位移 [u1,v1,u2,v2,u3,v3]：")
        print(np.round(d,6))
        print("支座约束反力：", np.round(f_react,4))
        # 后处理：单元应力轴力
        print("\n===== 各单元结果（长度L、方向余弦c,s、应力σ、轴力N）=====")
        for e in range(nel):
            L, c, s, E, A = elem_info[e]
            gdofs = LM[:,e]
            d_elem = d[gdofs]
            sigma, N = calc_stress_2d(d_elem, E, L, c, s)
            print(f"单元{e+1}: L={L:.4f}, c={c:.4f}, s={s:.4f}, σ={sigma:.6f}, N={N:.6f}")