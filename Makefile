env:
    @conda env update -f environment.yml --prune

html:
    @myst build --html

clean:
    @rm -rf figures audio _build