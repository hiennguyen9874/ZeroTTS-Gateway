import pytest

from zerotts_gateway.config import load_settings as load_gateway
from zerotts_server.config import load_settings as load_server


@pytest.mark.parametrize(
    ("load", "name", "default"),
    [
        (load_gateway, "ZEROTTS_GATEWAY_PORT", 8000),
        (load_server, "ZEROTTS_SERVER_PORT", 8080),
    ],
)
def test_port_env_overrides_config(load, name, default, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("port = 1234\n")
    assert load(env={}).port == default
    assert load(config, env={}).port == 1234
    assert load(config, env={name: "2345"}).port == 2345
    with pytest.raises(ValueError):
        load(config, env={name: "not-a-port"})
