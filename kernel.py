import torch


def ft_fupn(t: torch.Tensor, n: int = 1, nprod: int = 10) -> torch.Tensor:
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # 2^p sinc(t/2^p)
    p = torch.pow(2.0, torch.linspace(1, nprod, nprod, device=t.device))

    # First multiplier: sinc(0.5*t/pi)^n
    mult01 = torch.pow(torch.sinc(0.5 * t / torch.pi), n)

    # Second multiplier: product of sinc(t/(2^p * pi))
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)
    sinc_input = t_expanded / p_expanded / torch.pi
    mult02 = torch.prod(torch.sinc(sinc_input), dim=-1)

    return mult01 * mult02


def fupn(x: torch.Tensor, n: int = 3, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    n = 3
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    mlt = 2.0 / (n + 2)
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_fupn(mlt * torch.pi * idx, n, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(mlt * torch.pi * x_expanded * idx_expanded)

    out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=-1))

    # Apply support condition
    return torch.where(torch.abs(x) <= 1.0 / mlt, out, torch.zeros_like(x))


def ft_up(t: torch.Tensor, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{up}{(x)}

    :param t: real scalar or array
    :param nprod: integer scalar, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of 2: 2^p
    p = torch.pow(2.0, torch.linspace(1, nprod, nprod, device=t.device))

    # Compute product of sinc(t/(2^p * pi))
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)
    sinc_input = t_expanded / p_expanded / torch.pi
    out = torch.prod(torch.sinc(sinc_input), dim=-1)

    return out


def up(x: torch.Tensor, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{up}{(x)}

    :param x: real scalar or array
    :param nsum: integer scalar, nsum=100 by default
    :param nprod: integer scalar, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_up(torch.pi * idx, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

    out = 0.5 + torch.sum(coeff * cos_terms, dim=-1)

    # Apply support condition: support = [-1, 1]
    return torch.where(torch.abs(x) <= 1.0, out, torch.zeros_like(x))


def ft_upm(t: torch.Tensor, m: int = 1, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{up}_m{(x)}

    :param t: real scalar or array
    :param m: integer scalar, m=1 by default, m>=1 for appropriate computation
    :param nprod: integer scalar, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of 2*m
    p = torch.pow(2.0 * m, torch.linspace(1, nprod, nprod, device=t.device))

    # Prepare tensors for broadcasting
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)

    # Compute numerator: sinc(m*t/(p * pi))^2
    numerator_arg = m * t_expanded / (p_expanded * torch.pi)
    numerator = torch.sinc(numerator_arg) ** 2

    # Compute denominator: sinc(t/(p * pi))
    denominator_arg = t_expanded / (p_expanded * torch.pi)
    denominator = torch.sinc(denominator_arg)

    # Compute product along the last dimension
    return torch.prod(numerator / denominator, dim=-1)


def upm(x: torch.Tensor, m: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{up}_m{(x)}

    :param x: real scalar or array
    :param m: integer scalar, m=1 by default, m>=1 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_upm(torch.pi * idx, m, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

    out = 0.5 + torch.sum(coeff * cos_terms, dim=-1)

    # Apply support condition: support = [-1, 1]
    return torch.where(torch.abs(x) <= 1.0, out, torch.zeros_like(x))


def ft_ha(t: torch.Tensor, a: float = 2., nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{h}_a{(x)}

    :param t: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param nprod: integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of a
    p = torch.pow(a, torch.linspace(1, nprod, nprod, device=t.device))

    # Prepare tensors for broadcasting
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)

    # Compute sinc(t/(p * pi))
    arg = t_expanded / (p_expanded * torch.pi)

    # Compute product along the last dimension
    return torch.prod(torch.sinc(arg), dim=-1)


def ha(x: torch.Tensor, a: float = 2., nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{h}_a{(x)}

    :param x: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    mlt = a - 1

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_ha(mlt * torch.pi * idx, a, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

    out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=-1))

    # Apply support condition: support = [-1/mlt, 1/mlt]
    return torch.where(torch.abs(x) <= 1.0 / mlt, out, torch.zeros_like(x))


def ft_xin(t: torch.Tensor, n: int = 1, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{xi}_n{(x)}

    :param t: real scalar or array
    :param n: integer scalar, n=1 by default, n>=1 for appropriate computation
    :param nprod: integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of (n + 1)
    p = torch.pow(n + 1, torch.linspace(1, nprod, nprod, device=t.device))

    # Prepare tensors for broadcasting
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)

    # Compute sinc(t/(p * pi))^n
    arg = t_expanded / (p_expanded * torch.pi)

    # Compute product along the last dimension
    return torch.prod(torch.sinc(arg) ** n, dim=-1)


def xin(x: torch.Tensor, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{xi}_n{(x)}

    :param x: real scalar or array
    :param n: integer scalar, n=1 by default, n>=1 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_xin(torch.pi * idx, n, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

    out = 0.5 + torch.sum(coeff * cos_terms, dim=-1)

    # Apply support condition: support = [-1, 1]
    return torch.where(torch.abs(x) <= 1.0, out, torch.zeros_like(x))


def ft_chan(t: torch.Tensor, a: float = 2., n: int = 1, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{ch}_{a,n}{(x)}

    :param t: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=1 for appropriate computation
    :param nprod: integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of a
    p = torch.pow(a, torch.linspace(1, nprod, nprod, device=t.device))

    # Prepare tensors for broadcasting
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)

    # Compute sinc(t/(p * pi))^n
    arg = t_expanded / (p_expanded * torch.pi)

    # Compute product along the last dimension
    return torch.prod(torch.sinc(arg) ** n, dim=-1)


def chan(x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{ch}_{a,n}{(x)}

    :param x: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=1 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    mlt = (a - 1) / n

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_chan(mlt * torch.pi * idx, a, n, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

    out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=-1))

    # Apply support condition: support = [-1/mlt, 1/mlt]
    return torch.where(torch.abs(x) <= 1.0 / mlt, out, torch.zeros_like(x))


def ft_fipan(t: torch.Tensor, a: float = 2., n: int = 1, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{fip}_{a,n}{(x)}

    :param t: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=0 for appropriate computation
    :param nprod: integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Create powers of a
    p = torch.pow(a, torch.linspace(1, nprod, nprod, device=t.device))

    # Compute first multiplier: sinc(0.5 * t / pi)^n
    mult01 = torch.pow(torch.sinc(0.5 * t / torch.pi), n)

    # Compute second multiplier: product of sinc(t/(p * pi))
    t_expanded = t.unsqueeze(-1)
    p_expanded = p.unsqueeze(0)
    sinc_input = t_expanded / (p_expanded * torch.pi)
    mult02 = torch.prod(torch.sinc(sinc_input), dim=-1)

    return mult01 * mult02


def fipan(x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{fip}_{a,n}{(x)}

    :param x: real scalar or array
    :param a: real scalar, a=2 by default, a>1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=0 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    l = n + 2.0 / (a - 1.0)
    mlt = 2.0 / l

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_fipan(mlt * torch.pi * idx, a, n, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

    out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=-1))

    # Apply support condition: support = [-1/mlt, 1/mlt]
    return torch.where(torch.abs(x) <= 1.0 / mlt, out, torch.zeros_like(x))


def ft_fpmn(t: torch.Tensor, m: int = 2, n: int = 1, nprod: int = 10) -> torch.Tensor:
    r"""Fourier transform of atomic function \mathrm{fp}_{m,n}{(x)}

    :param t: real scalar or array
    :param m: integer scalar, m=2 by default, m>=1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=0 for appropriate computation
    :param nprod: integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')

    t = t.unsqueeze(-1) if t.dim() == 0 else t

    # Compute first multiplier: sinc(0.5 * t / pi)^n
    mult01 = torch.pow(torch.sinc(0.5 * t / torch.pi), n)

    # Compute second multiplier: ft_upm(t, m, nprod)
    mult02 = ft_upm(t, m, nprod)

    return mult01 * mult02


def fpmn(x: torch.Tensor, m: int = 2, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
    r"""Fourier series of atomic function \mathrm{fp}_{m,n}{(x)}

    :param x: real scalar or array
    :param m: integer scalar, m=2 by default, m>=1 for appropriate computation
    :param n: integer scalar, n=1 by default, n>=0 for appropriate computation
    :param nsum: nsum is an integer, nsum=100 by default
    :param nprod: nprod is an integer, nprod=10 by default,
                  nprod>=5 for appropriate computation
    """
    if nprod < 5:
        raise Exception('nprod must be greater or equal to 5')
    if nsum < 1:
        raise Exception('nsum must be greater than 0')

    x = x.unsqueeze(-1) if x.dim() == 0 else x

    mlt = 2.0 / (n + 2.0)

    # Create indices
    idx = torch.linspace(1, nsum, nsum, device=x.device)

    # Compute Fourier coefficients
    coeff = ft_fpmn(mlt * torch.pi * idx, m, n, nprod)

    # Compute Fourier series
    x_expanded = x.unsqueeze(-1)
    idx_expanded = idx.unsqueeze(0)
    cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

    out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=-1))

    # Apply support condition: support = [-1/mlt, 1/mlt]
    return torch.where(torch.abs(x) <= 1.0 / mlt, out, torch.zeros_like(x))