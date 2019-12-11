"""Generate binary proto file from text format."""

def text_to_bin(name, src, out, proto_name, proto_file):
    """Convert a text proto file to binary file.

    Args:
        name: name of the rule.
        src: the text file to convert.
        out: target output filename.
        proto_name: name of the proto.
        proto_file: the .proto file that contain the definition of proto_name.
    """

    native.genrule(
        name = name,
        srcs = [
            src,
            proto_file,
        ],
        outs = [out],
        cmd = ("$(locations @com_google_protobuf//:protoc)" +
               " --encode=" + proto_name +
               " $(location " + proto_file + ")" +
               " < $(location " + src + ")" +
               " > $@"),
        exec_tools = ["@com_google_protobuf//:protoc"],
    )
