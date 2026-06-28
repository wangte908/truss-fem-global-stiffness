import numpy as np
import json
import time
from scipy.sparse import csr_matrix
try:
    from pypardiso import spsolve
except ImportError:
    print("提示：未安装pypardiso，pardiso稀疏求解功能不可用，仅LDLT稠密求解正常运行")

# ====================== 2.3 原有全部基础函数（无需修改） ======================
def build_LM(IEN, nnp, nel, ndof):
    LM = np.zeros((2*ndof, nel), dtype=int)
    for e in range(nel):
        n1 = IEN[e][0] - 1
        n2 = IEN[e][1] - 1
        for d in range(ndof):
            LM[d, e] = n1 * ndof + d
            LM[d + ndof, e] = n2 * ndof + d
    return LM

def element_1d_bar(E, A, L):
    ke = (E * A / L) * np.array([[1, -1], [-1, 1]])
    return ke

def element_2d_truss(x1, y1, x2, y2, E, A):
    dx = x2 - x1
    dy = y2 - y1
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

def assemble_global(K, ke, LM, e):
    ndof_e = ke.shape[0]
    for a in range(ndof_e):
        for b in range(ndof_e):
            i = LM[a, e]
            j = LM[b, e]
            K[i, j] += ke[a, b]
    return K

def solve_reduction(K, f, fixed_dof, fixed_val):
    ndof_total = K.shape[0]
    all_dof = np.arange(ndof_total)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    KFF = K[np.ix_(free_dof, free_dof)]
    KEF = K[np.ix_(free_dof, fixed_dof)]
    KFE = KEF.T
    KEE = K[np.ix_(fixed_dof, fixed_dof)]
    dE = np.array(fixed_val)
    fF = f[free_dof]
    rhs = fF - KEF @ dE
    dF = np.linalg.solve(KFF, rhs)
    d = np.zeros(ndof_total)
    d[fixed_dof] = dE
    d[free_dof] = dF
    fE = KEE @ dE + KFE @ dF
    return d, fE, free_dof

def calc_stress_2d(d_elem, E, A, L, c, s):
    sigma = E / L * (-c * d_elem[0] - s * d_elem[1] + c * d_elem[2] + s * d_elem[3])
    N = sigma * A
    return sigma, N

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ====================== 2.4 全新LDLT求解模块（无BUG） ======================
def ldlt_factor(K):
    n = K.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    A_mat = K.copy()
    for k in range(n):
        sum_ld = 0.0
        for j in range(k):
            sum_ld += L[k, j] * D[j] * L[k, j]
        D[k] = A_mat[k, k] - sum_ld
        if D[k] <= 1e-12:
            raise ValueError(f"LDLT分解失败：第{k}阶主元{D[k]:.2e}，矩阵奇异/非正定")
        for i in range(k + 1, n):
            sum_l = 0.0
            for j in range(k):
                sum_l += L[i, j] * D[j] * L[k, j]
            L[i, k] = (A_mat[i, k] - sum_l) / D[k]
    return L, D

def ldlt_solve(L, D, R):
    n = len(R)
    y = np.zeros(n)
    # 前代 L y = R
    for i in range(n):
        y[i] = R[i] - np.dot(L[i, :i], y[:i])
    # 对角缩放 D z = y
    z = y / D
    # 回代 L^T x = z
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = z[i] - np.dot(L[i+1:n, i], x[i+1:n])
    return x

def residual_norm(K, x, rhs):
    res = K @ x - rhs
    norm_res = np.linalg.norm(res)
    norm_rhs = np.linalg.norm(rhs)
    rel_res = norm_res / norm_rhs if norm_rhs > 1e-15 else 0.0
    return res, norm_res, rel_res

def solve_equilibrium(K_FF, rhs, method="ldlt"):
    t_start = time.time()
    if method == "ldlt":
        L, D = ldlt_factor(K_FF)
        x = ldlt_solve(L, D, rhs)
    elif method == "pardiso":
        K_csr = csr_matrix(K_FF)
        x = spsolve(K_csr, rhs)
    else:
        raise NotImplementedError("仅支持 ldlt / pardiso 两种求解器")
    t_cost = time.time() - t_start
    _, _, rel_r = residual_norm(K_FF, x, rhs)
    info = {"time": t_cost, "relative_residual": rel_r}
    return x, info

def solve_reduction_new(K, fixed_dof, fixed_val, f_vec, method="ldlt"):
    ndof_total = K.shape[0]
    all_dof = np.arange(ndof_total)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    KFF = K[np.ix_(free_dof, free_dof)]
    KEF = K[np.ix_(free_dof, fixed_dof)]
    KFE = KEF.T
    KEE = K[np.ix_(fixed_dof, fixed_dof)]
    dE = np.array(fixed_val)
    fF = f_vec[free_dof]
    rhs = fF - KEF @ dE
    dF, solve_info = solve_equilibrium(KFF, rhs, method=method)
    d = np.zeros(ndof_total)
    d[fixed_dof] = dE
    d[free_dof] = dF
    fE = KEE @ dE + KFE @ dF
    return d, fE, free_dof, solve_info

# ====================== 主程序入口（包含桁架算例+2.4测试算例） ======================
if __name__ == "__main__":
    # 切换算例：1=一维两杆 2=二维桁架
    case_id = 2

    if case_id == 1:
        print("================ 算例1：一维两单元杆结构 ================")
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
        for e in range(nel):
            n1 = IEN[e][0] - 1
            n2 = IEN[e][1] - 1
            L = abs(x[n2] - x[n1])
            ke = element_1d_bar(EA_L_list[e], 1, L)
            K = assemble_global(K, ke, LM, e)
        print("组装后总体刚度矩阵：")
        print(K)
        sym_err = np.linalg.norm(K - K.T)
        print(f"刚度矩阵对称误差：{sym_err:.2e}")
        f = np.zeros(ndof_total)
        for idx, dof in enumerate(force_dof):
            f[dof] = force_val[idx]
        # 调用2.4新版求解
        d, f_react, free_dof, solve_info = solve_reduction_new(K, fixed_dof, fixed_val, f, method="ldlt")
        print(f"\n求解耗时：{solve_info['time']:.6f} s")
        print(f"方程相对残差：{solve_info['relative_residual']:.2e}")
        print("全局节点位移 d：", d)
        print("支座约束反力 fE：", f_react)

    elif case_id == 2:
        print("================ 算例2：二维平面桁架结构 ================")
        data = load_json("case2_2d.json")
        nnp = data["nnp"]
        ndof = data["ndof"]
        nel = data["nel"]
        IEN = data["IEN"]
        x_coord = data["x"]
        y_coord = data["y"]
        E_list = data["E"]
        A_list = data["CArea"]
        fixed_dof = data["fixed_dof"]
        fixed_val = data["fixed_value"]
        force_dof = data["force_dof"]
        force_val = data["force_value"]
        ndof_total = nnp * ndof
        K = np.zeros((ndof_total, ndof_total))
        LM = build_LM(IEN, nnp, nel, ndof)
        for e in range(nel):
            n1 = IEN[e][0] - 1
            n2 = IEN[e][1] - 1
            x1, y1 = x_coord[n1], y_coord[n1]
            x2, y2 = x_coord[n2], y_coord[n2]
            E = E_list[e]
            A = A_list[e]
            ke, L, c, s = element_2d_truss(x1, y1, x2, y2, E, A)
            K = assemble_global(K, ke, LM, e)
        print("总体刚度矩阵：")
        print(K)
        sym_err = np.linalg.norm(K - K.T)
        print(f"刚度对称误差：{sym_err:.2e}")
        f = np.zeros(ndof_total)
        for i, dof in enumerate(force_dof):
            f[dof - 1] = force_val[i]
        d, f_react, free_dof, solve_info = solve_reduction_new(K, fixed_dof, fixed_val, f, method="ldlt")
        print(f"\n求解耗时：{solve_info['time']:.6f} s")
        print(f"方程相对残差：{solve_info['relative_residual']:.2e}")
        print("全局位移向量 d：")
        print(d)
        print("支座反力 fE：")
        print(f_react)
        # 单元应力后处理
        print("\n===== 各单元轴力与应力 =====")
        for e in range(nel):
            n1 = IEN[e][0] - 1
            n2 = IEN[e][1] - 1
            x1, y1 = x_coord[n1], y_coord[n1]
            x2, y2 = x_coord[n2], y_coord[n2]
            E = E_list[e]
            A = A_list[e]
            dx = x2 - x1
            dy = y2 - y1
            L = np.sqrt(dx**2 + dy**2)
            c = dx / L
            s = dy / L
            idx1 = n1 * 2
            idx2 = n2 * 2
            de = np.array([d[idx1], d[idx1+1], d[idx2], d[idx2+1]])
            sig, N = calc_stress_2d(de, E, A, L, c, s)
            print(f"单元{e+1}：应力={sig:.4f}，轴力={N:.4f}")

    # ====================== 2.4 独立测试算例（桁架运行完自动执行） ======================
    print("\n\n==================== 【2.4独立测试算例1：100阶三对角正定矩阵】 ====================")
    n_tri = 100
    K_tri = np.zeros((n_tri, n_tri))
    for i in range(n_tri):
        K_tri[i,i] = 2.0
        if i > 0:
            K_tri[i, i-1] = -1.0
            K_tri[i-1, i] = -1.0
    rhs_tri = np.ones(n_tri)
    t0 = time.time()
    L_tri, D_tri = ldlt_factor(K_tri)
    x_tri = ldlt_solve(L_tri, D_tri, rhs_tri)
    t1 = time.time()
    _, nr, rr = residual_norm(K_tri, x_tri, rhs_tri)
    print(f"n={n_tri} 求解耗时：{t1-t0:.4f} s，相对残差={rr:.2e}")
    print(f"解前5项数值：{x_tri[:5]}")

    print("\n\n==================== 【2.4独立测试算例2：非正定奇异矩阵报错检测】 ====================")
    K_bad = np.array([[1, 2], [2, 1]])
    try:
        L_test, D_test = ldlt_factor(K_bad)
    except ValueError as err:
        print("捕获预期异常：", err)

    print("\n\n==================== 【2.4独立测试算例3：病态矩阵误差分析】 ====================")
    K_ill = np.array([[1.0, 1.0], [1.0, 1.0001]])
    x_exact = np.array([1.0, 1.0])
    rhs_ill = K_ill @ x_exact
    L_ill, D_ill = ldlt_factor(K_ill)
    x_calc = ldlt_solve(L_ill, D_ill, rhs_ill)
    _, nr_ill, rr_ill = residual_norm(K_ill, x_calc, rhs_ill)
    rel_err = np.linalg.norm(x_calc - x_exact) / np.linalg.norm(x_exact)
    cond_num = np.linalg.cond(K_ill)
    print(f"矩阵条件数 cond(K) = {cond_num:.2e}")
    print(f"方程相对残差 = {rr_ill:.2e}")
    print(f"解相对误差 = {rel_err:.2e}")
    print(f"精确解：{x_exact}，数值求解解：{x_calc}")