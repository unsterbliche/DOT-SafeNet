# Nature-figure source preflight

The shared Python style and export module was checked with the `nature-figure`
static validator from `Yuan1z0825/nature-skills`.

- Python syntax: pass
- Arial/Helvetica/Liberation Sans configuration: pass
- Minimum detected text size: 5.5 pt, pass
- Rainbow/jet/hsv color maps: absent
- Editable SVG and TrueType PDF settings: pass
- SVG and PDF exports: pass
- TIFF export at 600 dpi: pass
- Row sampling: absent
- Simulated observations: absent
- Cross-backend plotting calls: absent

The validator reported one warning because final width is defined by each panel
or composition script rather than by the shared module. Complete figures use a
183-mm width; individual panels use 89 mm or 183 mm according to panel density.

All panel scripts import this shared module. The package-level validator completed
116 numerical and file checks without failure. Direct browser inspection of local
`file://` images was unavailable under the browser security policy. PNG decoding,
dimensions, non-background pixel density and SVG text nodes were checked
programmatically; final human inspection at manuscript size remains required.

