def get_git_version():
    """
    Use git describe to get version based on git tag.
    Save output to file
    """
    from subprocess import check_output

    output = check_output(
        [
            "git",
            "describe",
            "--dirty",
        ]
    ).decode()
    version = output[:-1]  # had trailing newline
    return version
