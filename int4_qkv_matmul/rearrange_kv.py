import torch

torch.manual_seed(42)


def combine_uint8_to_uint32_little_endian(byte0: int, byte1: int, byte2: int, byte3: int) -> int:
    """
    将 4 个 uint8 类型的数据按照小端存储组合为一个 uint32 类型的数据。

    Args:
        byte0: 最低位字节 (uint8)。
        byte1: 次低位字节 (uint8)。
        byte2: 次高位字节 (uint8)。
        byte3: 最高位字节 (uint8)。

    Returns:
        组合成的 uint32 类型的数据 (int)。
    """
    # 使用位运算进行组合
    uint32_value = (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0
    return uint32_value

def process_32_v(tensor_32):
    a = tensor_32[:, 0:8]
    b = tensor_32[:, 8:16]
    c = tensor_32[:, 16:24]
    d = tensor_32[:, 24:32]
    output_list = []
    for i in range(0, 8, 1):
        row = torch.cat((
            a[:, i:i+1],
            b[:, i:i+1],
            c[:, i:i+1],
            d[:, i:i+1]
        ), dim=1)
        
        output_list.append(row)
    return torch.cat(output_list, dim=1)



def process_32_k(tensor_32):
    """针对长度为 32 的 tensor 执行之前的逻辑"""
    a = tensor_32[:, 0:8]
    b = tensor_32[:, 8:16]
    c = tensor_32[:, 16:24]
    d = tensor_32[:, 24:32]

    output_list = []
    for i in range(0, 8, 2):
        row = torch.cat((
            a[:, i:i+2],
            b[:, i:i+2],
            c[:, i:i+2],
            d[:, i:i+2]
        ), dim=1)
    
        index  = [0, 2, 4, 6, 1, 3, 5, 7]
        row_shuffle = torch.zeros_like(row)
        for i in range(8):
            row_shuffle[:, i] = row[:,index[i]]
        
        output_list.append(row_shuffle)
    return torch.cat(output_list, dim=1)


def process_n_128_tensor_independent(input_tensor_n_128, process_32):
    """处理 shape 为 [n, 128] 的 tensor，对每 32 个数独立执行逻辑"""
    n = input_tensor_n_128.shape[0]
    output_list = []
    for i in range(n):
        row = input_tensor_n_128[i, :].reshape(1, 128)
        # 分成 4 个 32 个数的部分
        part1 = row[:, 0:32]
        part2 = row[:, 32:64]
        part3 = row[:, 64:96]
        part4 = row[:, 96:128]

        # 对每个部分独立执行 process_32
        processed_part1 = process_32(part1)
        processed_part2 = process_32(part2)
        processed_part3 = process_32(part3)
        processed_part4 = process_32(part4)
    
        # 将处理后的四部分在列维度上拼接起来
        combined_row = torch.cat((processed_part1, processed_part2, processed_part3, processed_part4), dim=1)
        
        output_list.append(combined_row)

    return torch.cat(output_list, dim=0)


def rerange_k(k):
    return process_n_128_tensor_independent(k, process_32_k)

def rerange_v(v):
    return process_n_128_tensor_independent(v, process_32_v)


def pack_uint4_to_uint8(input_tensor):
    """
    将 shape 为 [n, 128] 的 uint8 tensor 中每行相邻的两个 uint4 值打包成一个 uint8 (小端存储)。

    Args:
        input_tensor: shape 为 [n, 128]，dtype 为 torch.uint8，元素值在 0-15 (uint4 范围) 内的 tensor。

    Returns:
        shape 为 [n, 64]，dtype 为 torch.uint8 的 tensor，其中每两个原始 uint4 值被打包成一个 uint8 (小端存储)。
    """
    assert input_tensor.dtype == torch.uint8, "输入 tensor 的数据类型必须是 torch.uint8"
    assert torch.all((input_tensor >= 0) & (input_tensor <= 15)), "输入 tensor 的所有元素值必须在 uint4 范围内 (0-15)"
    assert input_tensor.shape[1] == 128, "输入 tensor 的形状必须是 [n, 128]"

    n_rows = input_tensor.shape[0]

    # 将 tensor reshape 为 [n, 64, 2]，方便按对处理
    reshaped_tensor = input_tensor.reshape(n_rows, 64, 2)

    # 提取低位 nibble (第一个元素) 和高位 nibble (第二个元素)
    low_nibbles = reshaped_tensor[:, :, 0]
    high_nibbles = reshaped_tensor[:, :, 1]

    # 打包 (注意顺序：高位左移，或运算低位)
    packed_tensor = (high_nibbles << 4) | low_nibbles

    return packed_tensor



"""
seqlen_k = 2
head_size = 128
key = torch.randint(0, 15, (seqlen_k, head_size), dtype=torch.uint8, device="cuda")

torch.set_printoptions(profile="full")
print("uint4 key: ", key)

key_rearrange = rerange_k(key)
print("rearrange key: ", key_rearrange)

key_int4 = pack_uint4_to_uint8(key_rearrange)
print("int4 key: ", key_int4)
torch.set_printoptions(profile="default")

print("\n")
print("--------------------------------------------------------------------------")
print("\n")

value = torch.randint(0, 15, (seqlen_k, head_size), dtype=torch.uint8, device="cuda")

torch.set_printoptions(profile="full")
print("uint4 value: \n")
print(value)

value_rearrange = rerange_v(value)
print("rearrange value: \n")
print(value_rearrange)

value_int4 = pack_uint4_to_uint8(value_rearrange)
print("int4 value: \n")
print(value_int4)

torch.set_printoptions(profile="default")


uint32_result = combine_uint8_to_uint32_little_endian(195, 225, 25, 206)
print(f"uint32 结果: {uint32_result}")
"""