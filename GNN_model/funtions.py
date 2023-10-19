"""
    自定义的message 和 reduce 函数
"""

def u_mul_e_udf(edges):
    return {"m": edges.src["h"] * edges.data["w"]}

def mean_udf(nodes):
    return {"h_N": nodes.mailbox["m"].mean(1)}