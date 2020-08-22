def get_git_version():
    """
    Use git describe to get version based on git tag.
    Save output to file
    """
    from subprocess import check_output
    output = check_output(['git', 'describe', '--dirty',]).decode()
    version = output[:-1] # had trailing newline

    with open('.version', 'w') as f:
        f.write(version)

    return version


def read_version():
    """
    Reads version at runtime
    """
    try:
        with open('.version', 'r') as f:
            version = f.read()
        return version
    except:
        return None
