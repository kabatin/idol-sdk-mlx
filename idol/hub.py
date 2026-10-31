"""Hugging Face access checks shared by the torch and MLX backends (no torch import)."""
_ACCESSIBLE_REPOS = set()


class EncoderAccessError(RuntimeError):
    """The Hugging Face account in use cannot download one of the gated image encoders."""


def require_encoder_access(repos):
    """Stop with an explanation, before anything large is downloaded, when the gated encoder repos are not readable.

    Network problems are not treated as missing access: the encoders may already be in the local cache."""
    import os

    from huggingface_hub import auth_check, get_token
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if os.environ.get('HF_HUB_OFFLINE', '0').lower() in ('1', 'true', 'yes'):
        return
    denied = []
    for repo in repos:
        if repo in _ACCESSIBLE_REPOS:
            continue
        try:
            auth_check(repo)
            _ACCESSIBLE_REPOS.add(repo)
        except (GatedRepoError, RepositoryNotFoundError):
            denied.append(repo)
        except Exception:
            pass
    if not denied:
        return
    pages = '\n'.join(f'  https://huggingface.co/{repo}' for repo in denied)
    if get_token() is None:
        reason = 'You are not logged in to Hugging Face, and these image encoders are gated:'
        steps = ('1. Accept the terms on each page above with your Hugging Face account.\n'
                 '2. Log in: run `hf auth login`, or set the HF_TOKEN environment variable to a token of that account.')
    else:
        reason = 'Your Hugging Face account does not have access yet to these gated image encoders:'
        steps = ('1. Accept the terms on each page above (approval can take a while).\n'
                 '2. Make sure the token in use (`hf auth whoami`) belongs to the account that accepted them.')
    raise EncoderAccessError(f'Idol needs two image encoders that are downloaded from Hugging Face.\n{reason}\n'
                             f'{pages}\n{steps}')
