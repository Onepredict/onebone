TAG=$1
TAG=`echo $TAG | sed -e "s/v//g"`

PACKAGE_NAME="oplib"

echo "A tag triggered build. $TAG"

VERSION=`python -c "import oplib; print(oplib.__version__)"`
echo "The version is $VERSION"

if [ "$VERSION" = "$TAG" ]; then
    echo "Version is correct! Deploy to local server."

else
    echo "Tag and version are not same. Check again."
    exit 1
fi;