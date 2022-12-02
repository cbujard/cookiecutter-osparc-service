"""
The function 'demo' in this sample is used as an interface to publish this
code as a service in osparc

Current version of .osparc expects under src/

your python code
your requirements (e.g. requirements.txt)

This will be re-configurable in future releases
"""

def demo(
    *, x: int, y: float = 3.14
) -> bool:
    """some demo doc"""
    return x*y>0
