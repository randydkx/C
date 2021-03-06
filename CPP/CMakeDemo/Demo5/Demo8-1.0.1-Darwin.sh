#!/bin/sh

# Display usage
cpack_usage()
{
  cat <<EOF
Usage: $0 [options]
Options: [defaults in brackets after descriptions]
  --help            print this message
  --version         print cmake installer version
  --prefix=dir      directory in which to install
  --include-subdir  include the Demo8-1.0.1-Darwin subdirectory
  --exclude-subdir  exclude the Demo8-1.0.1-Darwin subdirectory
  --skip-license    accept license
EOF
  exit 1
}

cpack_echo_exit()
{
  echo $1
  exit 1
}

# Display version
cpack_version()
{
  echo "Demo8 Installer Version: 1.0.1, Copyright (c) Humanity"
}

# Helper function to fix windows paths.
cpack_fix_slashes ()
{
  echo "$1" | sed 's/\\/\//g'
}

interactive=TRUE
cpack_skip_license=FALSE
cpack_include_subdir=""
for a in "$@"; do
  if echo $a | grep "^--prefix=" > /dev/null 2> /dev/null; then
    cpack_prefix_dir=`echo $a | sed "s/^--prefix=//"`
    cpack_prefix_dir=`cpack_fix_slashes "${cpack_prefix_dir}"`
  fi
  if echo $a | grep "^--help" > /dev/null 2> /dev/null; then
    cpack_usage
  fi
  if echo $a | grep "^--version" > /dev/null 2> /dev/null; then
    cpack_version
    exit 2
  fi
  if echo $a | grep "^--include-subdir" > /dev/null 2> /dev/null; then
    cpack_include_subdir=TRUE
  fi
  if echo $a | grep "^--exclude-subdir" > /dev/null 2> /dev/null; then
    cpack_include_subdir=FALSE
  fi
  if echo $a | grep "^--skip-license" > /dev/null 2> /dev/null; then
    cpack_skip_license=TRUE
  fi
done

if [ "x${cpack_include_subdir}x" != "xx" -o "x${cpack_skip_license}x" = "xTRUEx" ]
then
  interactive=FALSE
fi

cpack_version
echo "This is a self-extracting archive."
toplevel="`pwd`"
if [ "x${cpack_prefix_dir}x" != "xx" ]
then
  toplevel="${cpack_prefix_dir}"
fi

echo "The archive will be extracted to: ${toplevel}"

if [ "x${interactive}x" = "xTRUEx" ]
then
  echo ""
  echo "If you want to stop extracting, please press <ctrl-C>."

  if [ "x${cpack_skip_license}x" != "xTRUEx" ]
  then
    more << '____cpack__here_doc____'
The MIT License (MIT)

Copyright (c) 2013 Joseph Pan(http://hahack.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

____cpack__here_doc____
    echo
    while true
      do
        echo "Do you accept the license? [yn]: "
        read line leftover
        case ${line} in
          y* | Y*)
            cpack_license_accepted=TRUE
            break;;
          n* | N* | q* | Q* | e* | E*)
            echo "License not accepted. Exiting ..."
            exit 1;;
        esac
      done
  fi

  if [ "x${cpack_include_subdir}x" = "xx" ]
  then
    echo "By default the Demo8 will be installed in:"
    echo "  \"${toplevel}/Demo8-1.0.1-Darwin\""
    echo "Do you want to include the subdirectory Demo8-1.0.1-Darwin?"
    echo "Saying no will install in: \"${toplevel}\" [Yn]: "
    read line leftover
    cpack_include_subdir=TRUE
    case ${line} in
      n* | N*)
        cpack_include_subdir=FALSE
    esac
  fi
fi

if [ "x${cpack_include_subdir}x" = "xTRUEx" ]
then
  toplevel="${toplevel}/Demo8-1.0.1-Darwin"
  mkdir -p "${toplevel}"
fi
echo
echo "Using target directory: ${toplevel}"
echo "Extracting, please wait..."
echo ""

# take the archive portion of this file and pipe it to tar
# the NUMERIC parameter in this command should be one more
# than the number of lines in this header file
# there are tails which don't understand the "-n" argument, e.g. on SunOS
# OTOH there are tails which complain when not using the "-n" argument (e.g. GNU)
# so at first try to tail some file to see if tail fails if used with "-n"
# if so, don't use "-n"
use_new_tail_syntax="-n"
tail $use_new_tail_syntax +1 "$0" > /dev/null 2> /dev/null || use_new_tail_syntax=""

extractor="pax -r"
command -v pax > /dev/null 2> /dev/null || extractor="tar xf -"

tail $use_new_tail_syntax +170 "$0" | gunzip | (cd "${toplevel}" && ${extractor}) || cpack_echo_exit "Problem unpacking the Demo8-1.0.1-Darwin"

echo "Unpacking finished successfully"

exit 0
#-----------------------------------------------------------
#      Start of TAR.GZ file
#-----------------------------------------------------------;
� Bb 磔飇G鹕%裩艤-垾G鍔�4冫瓍昱喉滿㈨E皞{菇芤唤p籯0C颌/��€玄�

�
吋鑻RR{)-馝i:郴椲瑮`K蟅鼅`23�3�>3s壇鎏izi1I拻�8惦凷K憳S籬8塇馜2�S)岺	B悱NL�ST芔�(ZZ瑟4菄
[膓妆^?#rOM仳卑攍�溸嘧扨⿻I�=珑�菝?NH[;ox倜说NB>&跃�2/i^;��畖Q煦1跒�9"蝏泩3,_n2?鍥圉;=呚y橁勝鎻�:崮语鹌� 愚.c唅鍖M沩=槟坨胱u柜秏埱妀iL�6�'贵弝鷣-�1酞n抇褝7嚅{6v鵿K佐<郁B集躼iO�肒滊溣沓CY'Z蕎狠-bL袨?Z�毽窈H#N籫钾�9嫾j雚<7yM7汐緁艼
3&斯J墠櫿M鉗O邲�軿��:耱姪勹娱嶋ぞx~嵾蟻绦{蜷L�,|隲p猚盥谌葡�!灩鋷项Zh|骓<溝�^媞"尭?消;D7	Y狨紲s�<ッ�~籀莖7q鰒/}朡�+i咕黡)镛磐悖s_�z鼖煹�8鹕73烥躽�w貳礉郊�!午_繄#⺗x墮掻{脍ot埁牍�2r鬶o笚'ュ祚羌闵丯�轘�踜n>�滮蓓孆                           �;围gI宵.輵夘煸jz諾櫃Zm锞~恄H仙+�9)}嚰w冊句,夼\委樿猐銤{錃{%X�*蔏|睥3w�攌sSK祱;~压b-駔p�㑇晕8Y蜕徱3伐K�
髇続1suv陙樇z炕g餉�%懲添啷[髃ˉ!k鱿汖蠓�5pe掼袭卬蝾|Q\\痉抖V瘇.�(Z忮襄顴氚h蒎�2IC�)咼諌眾&軮�*鉻\�縥Uj槉濿獃Z襯U�:偎噭F閑蔛蜖⊙濚{蔟髦鈣S颃j�8e.竓嬿^�傲芠縊蜈>                               <)��{?阒胣}�$�                                鄆賡�孃臫~矓g唅錢N芋j暭_%竽鄔榖V
⑤宕5�蝣Xe\4鲏fU覯1ó崙N`�"S'�薚YQUD貎eE覫(狼磽叓铕叓7殳x嚄菁祌蛮䜩烡繈:眙糯疃狊^鲒}BR铠魼=岰峻_�
_?铍w鹨伤6^扼艺錃@	k睼f�?g珈=謈[呜艁(懮k騊謱2>�US9�j鎮�炸a婷Q蚡JU3媏赵F2肔栥W敀r_Z 酸(搖%WR3匧y瑪蔎�8�)U�S蚫#,eWdY帣揷魕fv�<�+姪䲟O�i鶫墒珆瓅�$I蓌溭u漏泰]4婦"橪啯嶪�0●V&Ug濠尗篞创扷i>�+秷惝c絶F韵あ囱辀+灹�#媘q迄嚀~X"Jㄔ奷鼮篌镫-䁖O仦鲜l鹈窿�4黹垸_�  泓€5 �  