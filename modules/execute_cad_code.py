def execute_cad_code(code: str, result_var: str = "cad"):
    """
    Executes the given generated Python code in a controlled scope
    and returns the object assigned to `result_var`.

    Args:
        code (str): The Python code to execute.
        result_var (str): The name of the variable to extract from the execution scope.

    Returns:
        object: The value assigned to `result_var` in the code, or None.
    """
    # Prepare a clean namespace with only safe built-ins and required classes
    safe_globals = {
        "__builtins__": {
            "__import__": __import__,
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "float": float,
            "int": int,
            "print": print,
        },
    }

    local_vars = {}

    try:
        exec(code, safe_globals, local_vars)
        return local_vars.get(result_var)
    except Exception as e:
        print("❌ Error executing generated code:", e)
        return str(e)
