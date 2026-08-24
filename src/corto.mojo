"""Compute kernels ported from corto's mesh and point-cloud codec."""

from std.algorithm import parallelize
from std.ffi import external_call
from std.math import isfinite, sqrt
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_QUANTIZE_ELEMENTS = 1_048_576
comptime MAX_QUANTIZE_WORKERS = 7


@always_inline
def u8p(address: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=address)


@always_inline
def i32p(address: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=address)


@always_inline
def u32p(address: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=address)


@always_inline
def f32p(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


@always_inline
def i64p(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


@export("corto_parallel_init")
def corto_parallel_init() abi("C") -> Int:
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


# corto: include/corto/vertex_attribute.h GenericAttr::quantize
@export("corto_quantize_f32")
def corto_quantize_f32(
    source_address: Int,
    target_address: Int,
    count: Int,
    components: Int,
    q: Float32,
    offsets_address: Int,
    status_address: Int,
) abi("C") -> Int:
    if count <= 0 or components <= 0 or q == 0.0:
        return 0
    var source = f32p(source_address)
    var target = i32p(target_address)
    var offsets = f32p(offsets_address)
    var size = count * components
    for c in range(components):
        if not isfinite(offsets[c]):
            return 1
    if components != 3:
        for i in range(count):
            for c in range(components):
                var value = source[i * components + c]
                if not isfinite(value):
                    return 1
                var scaled = (value - offsets[c]) / q
                if scaled < -2147483648.0 or scaled >= 2147483648.0:
                    return 2
                target[i * components + c] = Int32(scaled)
        return 0

    comptime W = simdwidthof[DType.float64]()
    comptime BLOCK = 3 * W
    var low = SIMD[DType.float32, W](-2147483648.0)
    var high = SIMD[DType.float32, W](2147483648.0)
    var workers = min(num_physical_cores(), MAX_QUANTIZE_WORKERS)
    if size < PARALLEL_QUANTIZE_ELEMENTS:
        workers = 1

    @parameter
    def process(worker: Int):
        var work_source = f32p(source_address)
        var work_target = i32p(target_address)
        var work_offsets = f32p(offsets_address)
        var work_status = i32p(status_address)
        work_status[worker] = 0
        var blocks = size // BLOCK
        var start = (worker * blocks // workers) * BLOCK
        var end = ((worker + 1) * blocks // workers) * BLOCK
        if worker == workers - 1:
            end = size
        var offset0 = SIMD[DType.float32, W](0.0)
        var offset1 = SIMD[DType.float32, W](0.0)
        var offset2 = SIMD[DType.float32, W](0.0)
        for lane in range(W):
            offset0[lane] = work_offsets[lane % 3]
            offset1[lane] = work_offsets[(lane + W) % 3]
            offset2[lane] = work_offsets[(lane + 2 * W) % 3]
        var divisor = SIMD[DType.float32, W](q)
        var i = start
        while i + BLOCK <= end:
            var values0 = work_source.load[width=W](i)
            if not isfinite(values0).reduce_and():
                work_status[worker] = 1
            else:
                var scaled0 = (values0 - offset0) / divisor
                if scaled0.lt(low).reduce_or() or scaled0.ge(high).reduce_or():
                    if work_status[worker] == 0:
                        work_status[worker] = 2
                else:
                    work_target.store(i, scaled0.cast[DType.int32]())
            var values1 = work_source.load[width=W](i + W)
            if not isfinite(values1).reduce_and():
                work_status[worker] = 1
            else:
                var scaled1 = (values1 - offset1) / divisor
                if scaled1.lt(low).reduce_or() or scaled1.ge(high).reduce_or():
                    if work_status[worker] == 0:
                        work_status[worker] = 2
                else:
                    work_target.store(i + W, scaled1.cast[DType.int32]())
            var values2 = work_source.load[width=W](i + 2 * W)
            if not isfinite(values2).reduce_and():
                work_status[worker] = 1
            else:
                var scaled2 = (values2 - offset2) / divisor
                if scaled2.lt(low).reduce_or() or scaled2.ge(high).reduce_or():
                    if work_status[worker] == 0:
                        work_status[worker] = 2
                else:
                    work_target.store(i + 2 * W, scaled2.cast[DType.int32]())
            i += BLOCK
        while i < end:
            var value = work_source[i]
            if not isfinite(value):
                work_status[worker] = 1
            else:
                var scaled = (value - work_offsets[i % 3]) / q
                if scaled < -2147483648.0 or scaled >= 2147483648.0:
                    if work_status[worker] == 0:
                        work_status[worker] = 2
                else:
                    work_target[i] = Int32(scaled)
            i += 1

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
    var statuses = i32p(status_address)
    for worker in range(workers):
        if statuses[worker] == 1:
            return 1
    for worker in range(workers):
        if statuses[worker] == 2:
            return 2
    return 0


# corto: include/corto/vertex_attribute.h GenericAttr::dequantize
@export("corto_dequantize_f32")
def corto_dequantize_f32(
    source_address: Int,
    target_address: Int,
    count: Int,
    components: Int,
    q: Float32,
    offsets_address: Int,
) abi("C"):
    if count <= 0 or components <= 0:
        return
    var source = i32p(source_address)
    var target = f32p(target_address)
    var offsets = f32p(offsets_address)
    for i in range(count):
        for c in range(components):
            target[i * components + c] = (
                Float32(source[i * components + c]) * q + offsets[c]
            )


# corto: include/corto/vertex_attribute.h GenericAttr::deltaEncode
@export("corto_delta_encode")
def corto_delta_encode(
    values_address: Int,
    context_address: Int,
    diffs_address: Int,
    count: Int,
    components: Int,
    parallelogram: Int,
) abi("C"):
    if count <= 0 or components <= 0:
        return
    var values = i32p(values_address)
    var context = i32p(context_address)
    var diffs = i32p(diffs_address)
    for i in range(count):
        var t = Int(context[i * 4])
        var a = Int(context[i * 4 + 1])
        var b = Int(context[i * 4 + 2])
        var cidx = Int(context[i * 4 + 3])
        for c in range(components):
            if i == 0:
                diffs[c] = values[t * components + c]
            elif parallelogram != 0 and a != b:
                diffs[i * components + c] = values[t * components + c] - (
                    values[a * components + c]
                    + values[b * components + c]
                    - values[cidx * components + c]
                )
            else:
                diffs[i * components + c] = (
                    values[t * components + c] - values[a * components + c]
                )


# corto: include/corto/vertex_attribute.h GenericAttr::deltaDecode
@export("corto_delta_decode")
def corto_delta_decode(
    values_address: Int,
    context_address: Int,
    count: Int,
    components: Int,
    parallelogram: Int,
    point_cloud: Int,
) abi("C"):
    if count <= 1 or components <= 0:
        return
    var values = i32p(values_address)
    if point_cloud != 0:
        for i in range(1, count):
            for c in range(components):
                values[i * components + c] += values[(i - 1) * components + c]
        return
    var context = i32p(context_address)
    for i in range(1, count):
        var a = Int(context[i * 3])
        var b = Int(context[i * 3 + 1])
        var cidx = Int(context[i * 3 + 2])
        for c in range(components):
            if parallelogram != 0:
                values[i * components + c] += (
                    values[a * components + c]
                    + values[b * components + c]
                    - values[cidx * components + c]
                )
            else:
                values[i * components + c] += values[a * components + c]


@always_inline
def needed(value: Int32) -> Int:
    var a = Int(value)
    if a == 0:
        return 0
    if a == -1:
        return 1
    if a < 0:
        a = -a - 1
    var n = 2
    while (a >> 1) != 0:
        a >>= 1
        n += 1
    return n


# corto: include/corto/cstream.h OutStream::encodeArray
@export("corto_pack_correlated")
def corto_pack_correlated(
    values_address: Int,
    count: Int,
    components: Int,
    logs_address: Int,
    words_address: Int,
) abi("C") -> Int:
    if count <= 0 or components <= 0:
        return 0
    var values = i32p(values_address)
    var logs = u8p(logs_address)
    var words = u32p(words_address)
    var word_count = 0
    var buff = UInt32(0)
    var free_bits = 32
    for i in range(count):
        var diff = needed(values[i * components])
        for c in range(1, components):
            diff = max(diff, needed(values[i * components + c]))
        logs[i] = UInt8(diff)
        if diff == 0:
            continue
        var middle = Int64(1) << Int64(diff - 1)
        for c in range(components):
            var value = UInt32(Int64(values[i * components + c]) + middle)
            var nbits = diff
            if nbits >= free_bits:
                buff = (buff << UInt32(free_bits)) | (
                    value >> UInt32(nbits - free_bits)
                )
                words[word_count] = buff
                word_count += 1
                nbits -= free_bits
                if nbits > 0:
                    value &= UInt32((UInt64(1) << UInt64(nbits)) - 1)
                else:
                    value = 0
                free_bits = 32
                buff = 0
            if nbits > 0:
                buff = (buff << UInt32(nbits)) | value
                free_bits -= nbits
    if free_bits != 32:
        words[word_count] = buff << UInt32(free_bits)
        word_count += 1
    return word_count


# corto: include/corto/cstream.h InStream::decodeArray
@export("corto_unpack_correlated")
def corto_unpack_correlated(
    words_address: Int,
    word_count: Int,
    logs_address: Int,
    target_address: Int,
    count: Int,
    components: Int,
) abi("C") -> Int:
    if count <= 0 or components <= 0:
        return 0
    var words = u32p(words_address)
    var logs = u8p(logs_address)
    var target = i32p(target_address)
    var word_pos = 0
    var buff = UInt32(0)
    var available = 0
    for i in range(count):
        var diff = Int(logs[i])
        if diff < 0 or diff > 32:
            return -1
        if diff == 0:
            for c in range(components):
                target[i * components + c] = 0
            continue
        var middle = Int64(1) << Int64(diff - 1)
        for c in range(components):
            var result: UInt32
            if diff > available:
                if word_pos >= word_count:
                    return -1
                var high_bits = diff - available
                result = buff << UInt32(high_bits)
                buff = words[word_pos]
                word_pos += 1
                available = 32 - high_bits
                result |= buff >> UInt32(available)
                if available == 0:
                    buff = 0
                else:
                    buff &= UInt32((UInt64(1) << UInt64(available)) - 1)
            else:
                available -= diff
                result = buff >> UInt32(available)
                if available == 0:
                    buff = 0
                else:
                    buff &= UInt32((UInt64(1) << UInt64(available)) - 1)
            target[i * components + c] = Int32(Int64(result) - middle)
    return word_pos


# corto: include/corto/normal_attribute.h NormalAttr::toOcta
@export("corto_normals_to_octa")
def corto_normals_to_octa(
    normals_address: Int,
    target_address: Int,
    count: Int,
    bits: Int,
) abi("C"):
    if count <= 0:
        return
    var normals = f32p(normals_address)
    var target = i32p(target_address)
    var unit = Float32(Int64(1) << Int64(bits - 1))
    for i in range(count):
        var x = normals[i * 3]
        var y = normals[i * 3 + 1]
        var z = normals[i * 3 + 2]
        var length = abs(x) + abs(y) + abs(z)
        if length == 0.0:
            target[i * 2] = 0
            target[i * 2 + 1] = 0
            continue
        var px = x / length
        var py = y / length
        if z < 0.0:
            var old_x = px
            px = 1.0 - abs(py)
            py = 1.0 - abs(old_x)
            if x < 0.0:
                px = -px
            if y < 0.0:
                py = -py
        target[i * 2] = Int32(px * unit)
        target[i * 2 + 1] = Int32(py * unit)


# corto: include/corto/normal_attribute.h NormalAttr::toSphere
@export("corto_octa_to_normals")
def corto_octa_to_normals(
    octa_address: Int,
    target_address: Int,
    count: Int,
    bits: Int,
) abi("C"):
    if count <= 0:
        return
    var octa = i32p(octa_address)
    var target = f32p(target_address)
    var unit = Float32(Int64(1) << Int64(bits - 1))
    for i in range(count):
        var vx = Float32(octa[i * 2])
        var vy = Float32(octa[i * 2 + 1])
        var x = vx
        var y = vy
        var z = unit - abs(vx) - abs(vy)
        if z < 0.0:
            x = (
                Float32(1.0) if vx > 0.0 else Float32(-1.0)
            ) * (unit - abs(vy))
            y = (
                Float32(1.0) if vy > 0.0 else Float32(-1.0)
            ) * (unit - abs(vx))
        var length = sqrt(x * x + y * y + z * z)
        if length == 0.0:
            target[i * 3] = 0.0
            target[i * 3 + 1] = 0.0
            target[i * 3 + 2] = 1.0
        else:
            target[i * 3] = x / length
            target[i * 3 + 1] = y / length
            target[i * 3 + 2] = z / length


# corto: src/normal_attribute.cpp estimateNormals
@export("corto_estimate_normals")
def corto_estimate_normals(
    positions_address: Int,
    faces_address: Int,
    target_address: Int,
    vertex_count: Int,
    face_count: Int,
    normalize: Int,
) abi("C"):
    if vertex_count <= 0:
        return
    var positions = f32p(positions_address)
    var faces = u32p(faces_address)
    var target = f32p(target_address)
    for i in range(vertex_count * 3):
        target[i] = 0.0
    for f in range(face_count):
        var a = Int(faces[f * 3])
        var b = Int(faces[f * 3 + 1])
        var c = Int(faces[f * 3 + 2])
        if (
            a < 0 or b < 0 or c < 0
            or a >= vertex_count or b >= vertex_count or c >= vertex_count
        ):
            continue
        var e0x = positions[b * 3] - positions[a * 3]
        var e0y = positions[b * 3 + 1] - positions[a * 3 + 1]
        var e0z = positions[b * 3 + 2] - positions[a * 3 + 2]
        var e1x = positions[c * 3] - positions[a * 3]
        var e1y = positions[c * 3 + 1] - positions[a * 3 + 1]
        var e1z = positions[c * 3 + 2] - positions[a * 3 + 2]
        var nx = e0y * e1z - e0z * e1y
        var ny = e0z * e1x - e0x * e1z
        var nz = e0x * e1y - e0y * e1x
        target[a * 3] += nx
        target[a * 3 + 1] += ny
        target[a * 3 + 2] += nz
        target[b * 3] += nx
        target[b * 3 + 1] += ny
        target[b * 3 + 2] += nz
        target[c * 3] += nx
        target[c * 3 + 1] += ny
        target[c * 3 + 2] += nz
    if normalize != 0:
        for i in range(vertex_count):
            var x = target[i * 3]
            var y = target[i * 3 + 1]
            var z = target[i * 3 + 2]
            var length = sqrt(x * x + y * y + z * z)
            if length < 0.00001:
                target[i * 3] = 0.0
                target[i * 3 + 1] = 0.0
                target[i * 3 + 2] = 1.0
            else:
                target[i * 3] = x / length
                target[i * 3 + 1] = y / length
                target[i * 3 + 2] = z / length


# corto: src/color_attribute.cpp ColorAttr::quantize
@export("corto_colors_to_ycc")
def corto_colors_to_ycc(
    colors_address: Int,
    target_address: Int,
    count: Int,
    components: Int,
    steps_address: Int,
) abi("C"):
    if count <= 0 or components < 3 or components > 4:
        return
    var colors = u8p(colors_address)
    var target = u8p(target_address)
    var steps = u8p(steps_address)
    for i in range(count):
        var r = UInt8(Int(colors[i * components]) // Int(steps[0]))
        var g = UInt8(Int(colors[i * components + 1]) // Int(steps[1]))
        var b = UInt8(Int(colors[i * components + 2]) // Int(steps[2]))
        target[i * components] = g
        target[i * components + 1] = b - g
        target[i * components + 2] = r - g
        if components == 4:
            target[i * components + 3] = UInt8(
                Int(colors[i * components + 3]) // Int(steps[3])
            )


# corto: src/color_attribute.cpp ColorAttr::dequantize
@export("corto_ycc_to_colors")
def corto_ycc_to_colors(
    ycc_address: Int,
    target_address: Int,
    count: Int,
    components: Int,
    target_components: Int,
    steps_address: Int,
) abi("C"):
    if count <= 0 or components < 3 or components > 4:
        return
    var ycc = u8p(ycc_address)
    var target = u8p(target_address)
    var steps = u8p(steps_address)
    for i in range(count):
        var y = ycc[i * components]
        var cb = ycc[i * components + 1]
        var cr = ycc[i * components + 2]
        target[i * target_components] = UInt8(cr + y) * steps[0]
        target[i * target_components + 1] = y * steps[1]
        target[i * target_components + 2] = UInt8(cb + y) * steps[2]
        if target_components == 4:
            var alpha = ycc[i * components + 3] if components == 4 else UInt8(255)
            target[i * target_components + 3] = alpha * steps[3]


# corto: src/tunstall.cpp Tunstall::getProbabilities
@export("corto_histogram_u8")
def corto_histogram_u8(
    data_address: Int,
    size: Int,
    counts_address: Int,
) abi("C"):
    var data = u8p(data_address)
    var counts = u32p(counts_address)
    for i in range(256):
        counts[i] = 0
    for i in range(size):
        counts[Int(data[i])] += 1


# corto: src/tunstall.cpp Tunstall::createDecodingTables2
@export("corto_tunstall_build")
def corto_tunstall_build(
    symbols_address: Int,
    probabilities_address: Int,
    symbol_count: Int,
    index_address: Int,
    lengths_address: Int,
    table_address: Int,
    queues_address: Int,
    starts_address: Int,
    table_capacity: Int,
) abi("C") -> Int:
    if symbol_count <= 0:
        return 0
    var symbols = u8p(symbols_address)
    var probabilities = u8p(probabilities_address)
    var index = i32p(index_address)
    var lengths = i32p(lengths_address)
    var table = u8p(table_address)
    var queues = u32p(queues_address)
    var starts = i32p(starts_address)
    if symbol_count == 1:
        if table_capacity < 1:
            return -1
        index[0] = 0
        lengths[0] = 1
        table[0] = symbols[0]
        return 1

    var end = 0
    var pos = 0
    var n_words = 0
    var count = 2
    var p0 = Int(probabilities[0]) << 8
    var p1 = Int(probabilities[1]) << 8
    var probability = (p0 * p0) >> 16
    var max_count = 255 // (symbol_count - 1)
    while probability > p1 and count < max_count:
        probability = (probability * p0) >> 16
        count += 1

    if count >= 16:
        if 1 + (symbol_count - 1) * count > table_capacity:
            return -1
        table[pos] = symbols[0]
        pos += 1
        for k in range(1, symbol_count):
            for _ in range(count - 1):
                table[pos] = symbols[0]
                pos += 1
            table[pos] = symbols[k]
            pos += 1
        starts[0] = Int32((count - 1) * symbol_count)
        for k in range(1, symbol_count):
            starts[k] = Int32(k)
        for col in range(count):
            for row in range(1, symbol_count):
                var destination = row + col * symbol_count
                if col == 0:
                    queues[destination] = UInt32(Int(probabilities[row]) << 8)
                else:
                    queues[destination] = UInt32(
                        (probability * (Int(probabilities[row]) << 8)) >> 16
                    )
                index[destination] = Int32(row * count - col)
                lengths[destination] = Int32(col + 1)
            if col == 0:
                probability = p0
            else:
                probability = (probability * p0) >> 16
        var first = (count - 1) * symbol_count
        queues[first] = UInt32(probability)
        index[first] = 0
        lengths[first] = Int32(count)
        n_words = 1 + count * (symbol_count - 1)
        end = count * symbol_count
    else:
        n_words = symbol_count
        for i in range(symbol_count):
            starts[i] = Int32(i)
            queues[end] = UInt32(Int(probabilities[i]) << 8)
            index[end] = Int32(pos)
            lengths[end] = 1
            end += 1
            if pos >= table_capacity:
                return -1
            table[pos] = symbols[i]
            pos += 1

    while n_words < 256:
        var best = 0
        var max_probability = UInt32(0)
        for i in range(symbol_count):
            var candidate = queues[Int(starts[i])]
            if candidate > max_probability:
                best = i
                max_probability = candidate
        var selected = Int(starts[best])
        var selected_probability = Int(queues[selected])
        var offset = Int(index[selected])
        var length = Int(lengths[selected])
        var row = 0
        while row < symbol_count:
            if end >= 512 or pos + length + 1 > table_capacity:
                return -1
            var p = Int(probabilities[row])
            queues[end] = UInt32(
                (selected_probability * (p << 8)) >> 16
            )
            index[end] = Int32(pos)
            lengths[end] = Int32(length + 1)
            end += 1
            for j in range(length):
                table[pos] = table[offset + j]
                pos += 1
            table[pos] = symbols[row]
            pos += 1
            if n_words + row == 255:
                break
            row += 1
        if row == symbol_count:
            starts[best] += Int32(symbol_count)
        n_words += symbol_count - 1

    var word = 0
    var queue_row = 0
    for i in range(end):
        if queue_row >= symbol_count:
            queue_row = 0
        if Int(starts[queue_row]) <= i:
            index[word] = index[i]
            lengths[word] = lengths[i]
            word += 1
        queue_row += 1
    return min(word, 256)


# corto: src/tunstall.cpp Tunstall::compress
@export("corto_tunstall_encode")
def corto_tunstall_encode(
    data_address: Int,
    input_size: Int,
    index_address: Int,
    lengths_address: Int,
    table_address: Int,
    word_count: Int,
    target_address: Int,
) abi("C") -> Int:
    if input_size <= 0:
        return 0
    var data = u8p(data_address)
    var index = i32p(index_address)
    var lengths = i32p(lengths_address)
    var table = u8p(table_address)
    var target = u8p(target_address)
    var input_pos = 0
    var target_pos = 0
    while input_pos < input_size:
        var found = -1
        var found_length = 0
        var remaining = input_size - input_pos
        for code in range(word_count):
            var length = Int(lengths[code])
            var compared = min(length, remaining)
            var matches = True
            for j in range(compared):
                if data[input_pos + j] != table[Int(index[code]) + j]:
                    matches = False
                    break
            if matches and length <= remaining and length > found_length:
                found = code
                found_length = length
            elif matches and remaining < length and found < 0:
                found = code
                found_length = length
        if found < 0:
            return -1
        target[target_pos] = UInt8(found)
        target_pos += 1
        input_pos += min(found_length, remaining)
    return target_pos


# corto: src/tunstall.cpp Tunstall::decompress
@export("corto_tunstall_decode")
def corto_tunstall_decode(
    encoded_address: Int,
    encoded_size: Int,
    index_address: Int,
    lengths_address: Int,
    table_address: Int,
    word_count: Int,
    target_address: Int,
    output_size: Int,
) abi("C") -> Int:
    if output_size <= 0:
        return 0
    var encoded = u8p(encoded_address)
    var index = i32p(index_address)
    var lengths = i32p(lengths_address)
    var table = u8p(table_address)
    var target = u8p(target_address)
    var target_pos = 0
    var encoded_pos = 0
    while target_pos < output_size and encoded_pos < encoded_size:
        var code = Int(encoded[encoded_pos])
        encoded_pos += 1
        if code >= word_count:
            return -1
        var length = min(Int(lengths[code]), output_size - target_pos)
        var start = Int(index[code])
        for j in range(length):
            target[target_pos + j] = table[start + j]
        target_pos += length
    return target_pos


@always_inline
def edge_less(
    v0: U32Ptr, v1: U32Ptr, left: Int, right: Int
) -> Bool:
    if v0[left] != v0[right]:
        return v0[left] < v0[right]
    return v1[left] < v1[right]


@always_inline
def swap_edge(
    v0: U32Ptr,
    v1: U32Ptr,
    face: I32Ptr,
    side: U8Ptr,
    inverted: U8Ptr,
    a: Int,
    b: Int,
):
    var u = v0[a]
    v0[a] = v0[b]
    v0[b] = u
    u = v1[a]
    v1[a] = v1[b]
    v1[b] = u
    var iv = face[a]
    face[a] = face[b]
    face[b] = iv
    var byte = side[a]
    side[a] = side[b]
    side[b] = byte
    byte = inverted[a]
    inverted[a] = inverted[b]
    inverted[b] = byte


def sift_edges(
    v0: U32Ptr,
    v1: U32Ptr,
    face: I32Ptr,
    side: U8Ptr,
    inverted: U8Ptr,
    root: Int,
    end: Int,
):
    var current = root
    while current * 2 + 1 <= end:
        var child = current * 2 + 1
        if child + 1 <= end and edge_less(v0, v1, child, child + 1):
            child += 1
        if not edge_less(v0, v1, current, child):
            return
        swap_edge(v0, v1, face, side, inverted, current, child)
        current = child


# corto: src/encoder.cpp buildTopology
def build_topology_impl(
    faces: U32Ptr,
    face_count: Int,
    opposite_face: I32Ptr,
    opposite_side: I32Ptr,
    edge_v0: U32Ptr,
    edge_v1: U32Ptr,
    edge_face: I32Ptr,
    edge_side: U8Ptr,
    edge_inverted: U8Ptr,
):
    var edge_count = face_count * 3
    for f in range(face_count):
        for s in range(3):
            opposite_face[f * 3 + s] = -1
            opposite_side[f * 3 + s] = -1
            var a = Int(faces[f * 3 + ((s + 1) % 3)])
            var b = Int(faces[f * 3 + ((s + 2) % 3)])
            var e = f * 3 + s
            edge_face[e] = Int32(f)
            edge_side[e] = UInt8(s)
            if a < b:
                edge_v0[e] = UInt32(a)
                edge_v1[e] = UInt32(b)
                edge_inverted[e] = 0
            else:
                edge_v0[e] = UInt32(b)
                edge_v1[e] = UInt32(a)
                edge_inverted[e] = 1

    if edge_count > 1:
        var root = edge_count // 2
        while root > 0:
            root -= 1
            sift_edges(
                edge_v0, edge_v1, edge_face, edge_side, edge_inverted,
                root, edge_count - 1,
            )
        var end = edge_count - 1
        while end > 0:
            swap_edge(
                edge_v0, edge_v1, edge_face, edge_side, edge_inverted, 0, end
            )
            end -= 1
            sift_edges(
                edge_v0, edge_v1, edge_face, edge_side, edge_inverted, 0, end
            )

    var previous = -1
    for e in range(edge_count):
        if (
            previous >= 0
            and edge_v0[e] == edge_v0[previous]
            and edge_v1[e] == edge_v1[previous]
            and edge_inverted[e] != edge_inverted[previous]
        ):
            var f = Int(edge_face[e])
            var s = Int(edge_side[e])
            var pf = Int(edge_face[previous])
            var ps = Int(edge_side[previous])
            if opposite_face[f * 3 + s] == -1 and opposite_face[pf * 3 + ps] == -1:
                opposite_face[f * 3 + s] = Int32(pf)
                opposite_side[f * 3 + s] = Int32(ps)
                opposite_face[pf * 3 + ps] = Int32(f)
                opposite_side[pf * 3 + ps] = Int32(s)
        else:
            previous = e


# corto: src/encoder.cpp buildTopology
@export("corto_build_topology")
def corto_build_topology(
    faces_address: Int,
    face_count: Int,
    opposite_face_address: Int,
    opposite_side_address: Int,
    edge_v0_address: Int,
    edge_v1_address: Int,
    edge_face_address: Int,
    edge_side_address: Int,
    edge_inverted_address: Int,
) abi("C"):
    if face_count <= 0:
        return
    build_topology_impl(
        u32p(faces_address),
        face_count,
        i32p(opposite_face_address),
        i32p(opposite_side_address),
        u32p(edge_v0_address),
        u32p(edge_v1_address),
        i32p(edge_face_address),
        u8p(edge_side_address),
        u8p(edge_inverted_address),
    )


@always_inline
def front_append(
    front_face: I32Ptr,
    front_side: U8Ptr,
    front_prev: I32Ptr,
    front_next: I32Ptr,
    front_deleted: U8Ptr,
    position: Int,
    face: Int,
    side: Int,
    previous: Int,
    following: Int,
):
    front_face[position] = Int32(face)
    front_side[position] = UInt8(side)
    front_prev[position] = Int32(previous)
    front_next[position] = Int32(following)
    front_deleted[position] = 0


# corto: src/encoder.cpp Encoder::encodeFaces
@export("corto_connectivity_encode")
def corto_connectivity_encode(
    faces_address: Int,
    face_count: Int,
    vertex_count: Int,
    opposite_face_address: Int,
    opposite_side_address: Int,
    encoded_address: Int,
    prediction_address: Int,
    clers_address: Int,
    splits_address: Int,
    visited_address: Int,
    faceorder_address: Int,
    delayed_address: Int,
    front_face_address: Int,
    front_side_address: Int,
    front_prev_address: Int,
    front_next_address: Int,
    front_deleted_address: Int,
    metadata_address: Int,
) abi("C") -> Int:
    var metadata = i64p(metadata_address)
    for i in range(5):
        metadata[i] = 0
    if face_count <= 0:
        return 0
    var faces = u32p(faces_address)
    var opposite_face = i32p(opposite_face_address)
    var opposite_side = i32p(opposite_side_address)
    var encoded = i32p(encoded_address)
    var prediction = i32p(prediction_address)
    var clers = u8p(clers_address)
    var splits = i32p(splits_address)
    var visited = u8p(visited_address)
    var faceorder = i32p(faceorder_address)
    var delayed = i32p(delayed_address)
    var front_face = i32p(front_face_address)
    var front_side = u8p(front_side_address)
    var front_prev = i32p(front_prev_address)
    var front_next = i32p(front_next_address)
    var front_deleted = u8p(front_deleted_address)
    for i in range(vertex_count):
        encoded[i] = -1
    for i in range(face_count):
        visited[i] = 0

    var current_face = 0
    var current_vertex = 0
    var last_index = 0
    var remaining_faces = face_count
    var delayed_count = 0
    var faceorder_count = 0
    var order = 0
    var front_count = 0
    var cler_count = 0
    var split_count = 0
    var new_edge = -1

    while remaining_faces > 0:
        if (
            new_edge == -1
            and order >= faceorder_count
            and delayed_count == 0
        ):
            while current_face < face_count and visited[current_face] != 0:
                current_face += 1
            if current_face == face_count:
                break
            var first_edge = front_count
            var split_mask = 0
            for k in range(3):
                var vertex = Int(faces[current_face * 3 + k])
                if vertex < 0 or vertex >= vertex_count:
                    return -1
                if encoded[vertex] != -1:
                    split_mask |= 1 << k
            if split_mask != 0:
                clers[cler_count] = 6
                splits[split_count] = Int32(split_mask)
                split_count += 1
            else:
                clers[cler_count] = 0
            cler_count += 1
            for k in range(3):
                var vertex = Int(faces[current_face * 3 + k])
                if encoded[vertex] != -1:
                    splits[split_count] = encoded[vertex]
                    split_count += 1
                else:
                    prediction[current_vertex * 4] = Int32(vertex)
                    prediction[current_vertex * 4 + 1] = Int32(last_index)
                    prediction[current_vertex * 4 + 2] = Int32(last_index)
                    prediction[current_vertex * 4 + 3] = Int32(last_index)
                    encoded[vertex] = Int32(current_vertex)
                    current_vertex += 1
                    last_index = vertex
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, current_face, 0, first_edge + 2, first_edge + 1,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, current_face, 1, first_edge, first_edge + 2,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, current_face, 2, first_edge + 1, first_edge,
            )
            front_count += 1
            visited[current_face] = 1
            current_face += 1
            remaining_faces -= 1
            continue

        var edge = -1
        if new_edge != -1:
            edge = new_edge
            new_edge = -1
        elif order < faceorder_count:
            edge = Int(faceorder[order])
            order += 1
        elif delayed_count > 0:
            delayed_count -= 1
            edge = Int(delayed[delayed_count])
        else:
            return -2
        if front_deleted[edge] != 0:
            continue

        var source_face = Int(front_face[edge])
        var source_side = Int(front_side[edge])
        var other_face = Int(opposite_face[source_face * 3 + source_side])
        var other_side = Int(opposite_side[source_face * 3 + source_side])
        if other_face == -1 or visited[other_face] != 0:
            clers[cler_count] = 4
            cler_count += 1
            continue
        var k2 = other_side
        var k0 = (k2 + 1) % 3
        var k1 = (k0 + 1) % 3
        var edge_previous = Int(front_prev[edge])
        var edge_next = Int(front_next[edge])
        var previous_face = Int(front_face[edge_previous])
        var previous_side = Int(front_side[edge_previous])
        var next_face = Int(front_face[edge_next])
        var next_side = Int(front_side[edge_next])
        var close_left = (
            Int(opposite_face[previous_face * 3 + previous_side]) == other_face
        )
        var close_right = (
            Int(opposite_face[next_face * 3 + next_side]) == other_face
        )
        new_edge = front_count

        if close_left and close_right:
            clers[cler_count] = 3
            cler_count += 1
            front_deleted[edge_previous] = 1
            front_deleted[edge_next] = 1
            var previous_previous = Int(front_prev[edge_previous])
            var next_next = Int(front_next[edge_next])
            front_next[previous_previous] = Int32(next_next)
            front_prev[next_next] = Int32(previous_previous)
            new_edge = -1
        elif close_left:
            clers[cler_count] = 1
            cler_count += 1
            front_deleted[edge_previous] = 1
            var previous_previous = Int(front_prev[edge_previous])
            front_next[previous_previous] = Int32(new_edge)
            front_prev[edge_next] = Int32(new_edge)
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, other_face, k1, previous_previous, edge_next,
            )
            front_count += 1
        elif close_right:
            clers[cler_count] = 2
            cler_count += 1
            front_deleted[edge_next] = 1
            var next_next = Int(front_next[edge_next])
            front_prev[next_next] = Int32(new_edge)
            front_next[edge_previous] = Int32(new_edge)
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, other_face, k0, edge_previous, next_next,
            )
            front_count += 1
        else:
            var v0 = Int(faces[other_face * 3 + k0])
            var v1 = Int(faces[other_face * 3 + k1])
            var opposite = Int(faces[other_face * 3 + k2])
            if encoded[opposite] != -1 and order < faceorder_count:
                delayed[delayed_count] = Int32(edge)
                delayed_count += 1
                clers[cler_count] = 5
                cler_count += 1
                new_edge = -1
                continue
            if encoded[opposite] != -1:
                clers[cler_count] = 6
                splits[split_count] = encoded[opposite]
                split_count += 1
            else:
                clers[cler_count] = 0
                prediction[current_vertex * 4] = Int32(opposite)
                prediction[current_vertex * 4 + 1] = Int32(v0)
                prediction[current_vertex * 4 + 2] = Int32(v1)
                prediction[current_vertex * 4 + 3] = Int32(
                    faces[source_face * 3 + source_side]
                )
                encoded[opposite] = Int32(current_vertex)
                current_vertex += 1
                last_index = opposite
            cler_count += 1
            front_next[edge_previous] = Int32(new_edge)
            front_prev[edge_next] = Int32(new_edge + 1)
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, other_face, k0, edge_previous, new_edge + 1,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            front_append(
                front_face, front_side, front_prev, front_next, front_deleted,
                front_count, other_face, k1, new_edge, edge_next,
            )
            front_count += 1
        visited[other_face] = 1
        remaining_faces -= 1

    metadata[0] = Int64(cler_count)
    metadata[1] = Int64(split_count)
    metadata[2] = Int64(current_vertex)
    metadata[3] = Int64(front_count)
    metadata[4] = Int64(face_count)
    return 0


@always_inline
def decode_front_append(
    v0: I32Ptr,
    v1: I32Ptr,
    v2: I32Ptr,
    previous: I32Ptr,
    following: I32Ptr,
    deleted: U8Ptr,
    position: Int,
    a: Int,
    b: Int,
    c: Int,
    p: Int,
    n: Int,
):
    v0[position] = Int32(a)
    v1[position] = Int32(b)
    v2[position] = Int32(c)
    previous[position] = Int32(p)
    following[position] = Int32(n)
    deleted[position] = 0


# corto: src/decoder.cpp Decoder::decodeFaces
@export("corto_connectivity_decode")
def corto_connectivity_decode(
    clers_address: Int,
    cler_count: Int,
    splits_address: Int,
    split_count: Int,
    vertex_count: Int,
    face_count: Int,
    faces_address: Int,
    prediction_address: Int,
    faceorder_address: Int,
    delayed_address: Int,
    front_v0_address: Int,
    front_v1_address: Int,
    front_v2_address: Int,
    front_prev_address: Int,
    front_next_address: Int,
    front_deleted_address: Int,
    metadata_address: Int,
) abi("C") -> Int:
    var metadata = i64p(metadata_address)
    for i in range(4):
        metadata[i] = 0
    if face_count <= 0:
        return 0
    var clers = u8p(clers_address)
    var splits = i32p(splits_address)
    var faces = u32p(faces_address)
    var prediction = i32p(prediction_address)
    var faceorder = i32p(faceorder_address)
    var delayed = i32p(delayed_address)
    var front_v0 = i32p(front_v0_address)
    var front_v1 = i32p(front_v1_address)
    var front_v2 = i32p(front_v2_address)
    var front_prev = i32p(front_prev_address)
    var front_next = i32p(front_next_address)
    var front_deleted = u8p(front_deleted_address)
    var vertex_pos = 0
    var face_pos = 0
    var cler_pos = 0
    var split_pos = 0
    var front_count = 0
    var faceorder_count = 0
    var order = 0
    var delayed_count = 0
    var new_edge = -1

    while face_pos < face_count:
        if (
            new_edge == -1
            and order >= faceorder_count
            and delayed_count == 0
        ):
            if cler_pos >= cler_count:
                return -5
            var last_index = vertex_pos - 1
            var split_mask = 0
            var symbol = Int(clers[cler_pos])
            cler_pos += 1
            if symbol == 6:
                if split_pos >= split_count:
                    return -6
                split_mask = Int(splits[split_pos])
                split_pos += 1
                if split_mask < 0 or split_mask > 7:
                    return -7
            elif symbol != 0:
                return -1
            var vertices0 = 0
            var vertices1 = 0
            var vertices2 = 0
            for k in range(3):
                var vertex = 0
                if (split_mask & (1 << k)) != 0:
                    if split_pos >= split_count:
                        return -6
                    vertex = Int(splits[split_pos])
                    split_pos += 1
                    if vertex < 0 or vertex >= vertex_count:
                        return -7
                else:
                    if vertex_pos >= vertex_count:
                        return -2
                    prediction[vertex_pos * 3] = Int32(last_index)
                    prediction[vertex_pos * 3 + 1] = Int32(last_index)
                    prediction[vertex_pos * 3 + 2] = Int32(last_index)
                    vertex = vertex_pos
                    vertex_pos += 1
                    last_index = vertex
                if k == 0:
                    vertices0 = vertex
                elif k == 1:
                    vertices1 = vertex
                else:
                    vertices2 = vertex
                faces[face_pos * 3 + k] = UInt32(vertex)
            face_pos += 1
            var first_edge = front_count
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, vertices1, vertices2, vertices0,
                first_edge + 2, first_edge + 1,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, vertices2, vertices0, vertices1,
                first_edge, first_edge + 2,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, vertices0, vertices1, vertices2,
                first_edge + 1, first_edge,
            )
            front_count += 1
            continue

        var edge = -1
        if new_edge != -1:
            edge = new_edge
            new_edge = -1
        elif order < faceorder_count:
            edge = Int(faceorder[order])
            order += 1
        elif delayed_count > 0:
            delayed_count -= 1
            edge = Int(delayed[delayed_count])
        else:
            return -3
        if front_deleted[edge] != 0:
            continue
        if cler_pos >= cler_count:
            return -5
        var symbol = Int(clers[cler_pos])
        cler_pos += 1
        if symbol == 4:
            continue
        var v0 = Int(front_v0[edge])
        var v1 = Int(front_v1[edge])
        var previous_edge = Int(front_prev[edge])
        var next_edge = Int(front_next[edge])
        new_edge = front_count
        var opposite = -1
        if symbol == 0 or symbol == 6:
            if symbol == 6:
                if split_pos >= split_count:
                    return -6
                opposite = Int(splits[split_pos])
                split_pos += 1
                if opposite < 0 or opposite >= vertex_count:
                    return -7
            else:
                if vertex_pos >= vertex_count:
                    return -2
                prediction[vertex_pos * 3] = Int32(v1)
                prediction[vertex_pos * 3 + 1] = Int32(v0)
                prediction[vertex_pos * 3 + 2] = front_v2[edge]
                opposite = vertex_pos
                vertex_pos += 1
            front_next[previous_edge] = Int32(new_edge)
            front_prev[next_edge] = Int32(new_edge + 1)
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, v0, opposite, v1,
                previous_edge, new_edge + 1,
            )
            front_count += 1
            faceorder[faceorder_count] = Int32(front_count)
            faceorder_count += 1
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, opposite, v1, v0,
                new_edge, next_edge,
            )
            front_count += 1
        elif symbol == 1:
            front_deleted[previous_edge] = 1
            var previous_previous = Int(front_prev[previous_edge])
            front_next[previous_previous] = Int32(new_edge)
            front_prev[next_edge] = Int32(new_edge)
            opposite = Int(front_v0[previous_edge])
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, opposite, v1, v0,
                previous_previous, next_edge,
            )
            front_count += 1
        elif symbol == 2:
            front_deleted[next_edge] = 1
            var next_next = Int(front_next[next_edge])
            front_prev[next_next] = Int32(new_edge)
            front_next[previous_edge] = Int32(new_edge)
            opposite = Int(front_v1[next_edge])
            decode_front_append(
                front_v0, front_v1, front_v2, front_prev, front_next,
                front_deleted, front_count, v0, opposite, v1,
                previous_edge, next_next,
            )
            front_count += 1
        elif symbol == 5:
            delayed[delayed_count] = Int32(edge)
            delayed_count += 1
            new_edge = -1
            continue
        elif symbol == 3:
            front_deleted[previous_edge] = 1
            front_deleted[next_edge] = 1
            var previous_previous = Int(front_prev[previous_edge])
            var next_next = Int(front_next[next_edge])
            front_next[previous_previous] = Int32(next_next)
            front_prev[next_next] = Int32(previous_previous)
            opposite = Int(front_v0[previous_edge])
            new_edge = -1
        else:
            return -4
        faces[face_pos * 3] = UInt32(v1)
        faces[face_pos * 3 + 1] = UInt32(v0)
        faces[face_pos * 3 + 2] = UInt32(opposite)
        face_pos += 1

    metadata[0] = Int64(cler_pos)
    metadata[1] = Int64(split_pos)
    metadata[2] = Int64(vertex_pos)
    metadata[3] = Int64(front_count)
    return 0
