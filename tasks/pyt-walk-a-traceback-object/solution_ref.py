import os


def walk_traceback(exc):
    result = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        result.append(
            (
                os.path.basename(frame.f_code.co_filename),
                tb.tb_lineno,
                frame.f_code.co_name,
            )
        )
        tb = tb.tb_next
    return result
