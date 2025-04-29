import zlib
from reedsolo import RSCodec
rs = RSCodec(250)


def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))

def byteString_to_bitString(bit_string):
    """Convert bit string of <255 to a string of bits 8 chars long"""
    # Remove the 0b of the bit string
    bit_string = bit_string[2:]
    # Append 0's to keep bit string 8 chars
    bit_string = '00000000'[len(bit_string):] + bit_string
    return bit_string


def bytearray_to_bits(byte_array):
    """Convert array of bytes to a list of bits"""
    bit_string = []
    for byte in byte_array:
        bits = byteString_to_bitString(bin(byte))
        bit_string.extend([int(b) for b in bits])
    return bit_string


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        byteString = ''.join([str(bit) for bit in byte])
        byteInteger = int(byteString, 2)
        ints.append(byteInteger)

    return bytearray(ints)


def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    # Encoding text into bytes and compressing for faster encoding
    byte_array = zlib.compress(text.encode("utf-8"))
    # Takes byte array then encodes using RS algorithm
    byte_array = rs.encode(bytearray(byte_array))
    return byte_array


def bytearray_to_text(byte_array):
    """Apply error correction and decompress"""
    try:
        # Returns the Reed solomon error corrected message
        text = rs.decode(byte_array)[0]
        # Decompressed data into bytes
        text = zlib.decompress(text)
        # Decodes bytes to text
        return text.decode("utf-8")
    except BaseException:
        return False


if __name__ == '__main__':
    original_message = "adsfg"*1000
    bits = text_to_bits(original_message)
    recovered_message = bits_to_text(bits)
    print("recovered message : " + recovered_message)
