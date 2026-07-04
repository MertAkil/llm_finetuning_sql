try:
    from jinja2 import Template
except ImportError:
    Template = None

try:
    from .prompts import USER_PROMPT
except ImportError:
    from prompts import USER_PROMPT


def create_system_template(system):
    system_template = [    
        {
            "role": "system",
            "content": system,
        },
    ]

    return system_template

def create_template(system, user):
    TEMPLATE = [
        # {
        #     "role": "system",
        #     "content": system,
        # },
        {
            "role": "user",
            "content": user,
        },
    ]
    
    return TEMPLATE

def create_message(question, context):
    if Template is not None:
        user_template = Template(USER_PROMPT).render(question=question, context=context)
    else:
        user_template = (
            USER_PROMPT
            .replace("{{ question }}", question)
            .replace("{{ context }}", context)
        )
    return user_template
