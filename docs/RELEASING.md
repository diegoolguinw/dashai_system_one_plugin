# Publicar una versión en PyPI

> **PyPI es inmutable:** un archivo subido no se puede borrar ni reemplazar, y un número de versión no se puede
> reutilizar (solo se puede "yank", que lo oculta de la resolución pero no lo elimina). Ensaya siempre en
> TestPyPI antes de publicar.

El workflow [`release.yml`](../.github/workflows/release.yml) publica con *trusted publishing* (OIDC): no hay
tokens guardados en el repositorio.

## 1. Configuración (una sola vez)

1. Crea cuenta en [pypi.org](https://pypi.org) y en [test.pypi.org](https://test.pypi.org) con **2FA** activado.
2. En cada una, ve a *Your account → Publishing → Add a new pending publisher* y rellena:

   | Campo | PyPI | TestPyPI |
   |---|---|---|
   | PyPI project name | `dashai-system-one-plugin` | `dashai-system-one-plugin` |
   | Owner / Repository | tu usuario u organización de GitHub y el nombre del repo | igual |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

3. En GitHub: *Settings → Environments*, crea `pypi` y `testpypi`. En `pypi` conviene añadir un revisor
   obligatorio para que nadie publique sin aprobación.
4. El nombre `dashai-system-one-plugin` estaba libre en PyPI y TestPyPI al preparar esta guía; compruébalo de
   nuevo antes del primer envío.

## 2. Antes de cada versión

1. Cambia `__version__` en `src/dashai_system_one_plugin/__init__.py` (es la **única** fuente de la versión).
2. En `CHANGELOG.md`, cambia `Sin publicar` por la fecha de la entrega.
3. Ejecuta las comprobaciones en un entorno limpio:

   ```bash
   python -m venv .venv-release && . .venv-release/bin/activate
   pip install -e ".[dev]"
   python scripts/check_release.py
   ```

   Construye sdist y wheel, ejecuta `twine check`, instala el wheel en un entorno nuevo, carga los 4 *entry
   points* y corre los tests contra el paquete instalado. No sube nada.

## 3. Ensayo en TestPyPI

En GitHub: *Actions → Release → Run workflow* (rama `main`). Después, en un entorno limpio:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ dashai-system-one-plugin==X.Y.Z
python -c "import dashai_system_one_plugin as p; print(p.__version__)"
```

(Un archivo de TestPyPI tampoco se puede reemplazar: si algo falla, sube una versión nueva, p. ej. `X.Y.Z.post1`
o un número mayor.)

## 4. Publicar en PyPI

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

El workflow comprueba que el tag coincida con `__version__`, construye, valida y publica (pide la aprobación del
entorno `pypi` si la configuraste). Luego verifica:

```bash
pip install dashai-system-one-plugin==X.Y.Z
```

## 5. Si algo sale mal
- **Versión publicada con un error grave:** *Manage → Yank* en PyPI y publica una versión corregida.
- **El workflow falla en "trusted publishing":** revisa que owner, repo, `release.yml` y el nombre del entorno
  coincidan exactamente con lo registrado en PyPI.
- **Alternativa sin OIDC:** `twine upload` con un *API token* limitado a este proyecto, que nunca debe
  guardarse en el repositorio.

## 6. Qué es público al publicar
El nombre y el correo de `authors` en `pyproject.toml`, el README (se muestra como descripción) y el código
fuente del sdist. Revisa que sea lo que quieres mostrar.
