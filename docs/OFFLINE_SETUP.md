# Offline dependency installation

The execution environment used for automated checks routes HTTP requests
through a proxy that blocks outbound connections to the Python Package Index.
As a result ``pip install pandas numpy`` fails with ``403 Forbidden`` errors.

To work around this limitation you can stage pre-downloaded wheel files for the
required scientific libraries and install them from the local filesystem:

1. Download compatible wheels for ``numpy`` and ``pandas`` on a machine with
   internet access.  For example, for Python 3.13 on Linux x86_64 you might
   obtain files similar to:

   ```text
   numpy-2.1.3-cp313-cp313-manylinux_2_28_x86_64.whl
   pandas-2.2.3-cp313-cp313-manylinux_2_28_x86_64.whl
   ```

2. Copy the wheel files into ``vendor/wheels`` inside the repository.

3. From the project root run:

   ```bash
   python -m pip install --no-index --find-links vendor/wheels numpy pandas
   ```

   The ``--no-index`` flag forces pip to ignore the blocked public index and
   install directly from the supplied wheel files.

Once the wheels are staged locally, ``pytest`` and other tooling that depends on
``numpy`` and ``pandas`` will be able to import the packages normally.
