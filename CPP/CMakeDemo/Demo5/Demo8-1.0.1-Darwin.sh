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
閿燂拷 椤欑嚇b 绾炬棃顥�G妤ｏ拷%鐟佲晞澧�-閸拷G闁告棑鎷�4閸愵偆鎼烽弰鍗炴瀱濠婅￥鍩丒閻拷{閼垮洩濮㈤崬顦栫猾锟�0C妫帮拷/椤氫緤鎷烽敓瑙ｅ亾閻滐拷閿燂拷

閿燂拷
閸氬鎳烺R{)-妫ｏ拷i:闁瓨銇掗悷鐢閿滃懘绱朻23閿燂拷3閿燂拷>3s婢瑰洭甯峣zi1I閹蜂紮鎷�8閹箑鍤狵閹茬爤缁拷8婵夊洭顪�2閿燂拷S)瀹€锟�	B閹洟L閿燂拷S椤儝閼烘棑鎷�(ZZ閻燂拷4閼匡拷
[閼舵挸顩玘?#r椤炲槣椤愵敵椤熸ü鎬€椤濆骸宕欓弨宥忔嫹濠хǹ妲ˇ鏍ㄥ⒕楗睂閿燂拷=閻濇埊鎷烽懣锟�?椤狀枖H[;ox閸婃粏顕㎞B>&鐠哄喛鎷�2/i^;閿熸枻鎷烽悾鏈￠褏鍙�1鐠烘帪鎷�9"閾﹀繑鐣�3,_n2?闁搞儱婀�;=閸涙濮椾礁瀚�闁逛紮鎷�:瀹曨噮宓佺拠顓㈢閿燂拷 閹帮拷.c閸炲懘宕燤濞岋拷=濡茬喎娼归懗鐪滈弻婊咁潝閸╁崬顩L閿燂拷6閿燂拷'鐠愰潧绱虫锟�-閿燂拷1闁扮儐娲縩閹跺洩顧�7閸ゅ巿6v姒堢竼娴ｏ拷<闁竻椤у繘娉﹂煬绯筄閿燂拷閼叉帗绮栭—鍏兼捣椤烆剦鎲板▽鎻匶'Z閽戝寒鍟庨悪锟�-bL鐞氾拷?Z閿燂拷濮ｇ晫鐛擧#N缁偊鎹�閿燂拷9鐎氱穮闂嗭拷<7椤╊柨M7濮规劗绌懝锟�
3&閺傜枦婢х姵顏糓闁存」闁拷閿熷€熻雹閿熸枻鎷�:閼板崬袠椤楊亜瀚绘繛鍗炵Ъ閵囩€瀹撴崘鐔佺紒顩￠摎绋敓锟�,|闂呯灓閻氭氨娲㈢拫宀冩喗閿燂拷!閻忊晠瀚�妞ょ瓰h|妤狅拷<濠ф繐鎷�^婵拷"鐏忥拷?濞戯拷;D7	Y閻欙拷缁辩灚閿燂拷<閵夊喛鎷�~缁偓閼撅拷7q姒橈拷/}閺堚槄鎷�+i閸滄洟寮�)闂€锟�绾炬劖鍊磗_閿熺挡姒ф牬澧烽悡鐧告嫹8妤ｏ拷73閻戙儴鑸敓锟�w鐠ㄥ磭顦柈锟�閿燂拷!閸楀牜鍣�_缁伙拷#澶傛婢ь喗骞�{閼村朝t閸╀胶澧犻敓锟�2r妤濈缁楋拷'閵夈儳顨熺紘宀勬娑擃垽鎷烽娆炬椤忓嫯缍愰敓锟�闊笜>閿燂拷濠婏拷閽冩挸鐡�                           閿燂拷;閸ョI鐎癸拷.鏉擄拷婢舵鍚€jz鐠滅偓鐝抁m闁跨€幁鍑ユ禒锟�+閿燂拷9)}椤炴牕姣巜閸愬﹤褰�,婢剁唱婵拷濡法瀵柕銈庡竸{闁峰剝%X閿燂拷*閽勫饥閻拷3w閿燂拷閺€瀹籗K缁侊拷;~閸樺獙-妞愭攽閿燂拷閵橈拷閺咃拷8Y閾氭洖棰�3娴兼€熼敓锟�
妤傚洨绋�1suv闂勬瑦鈻弞閻愭槹妞佸鎷�%閹冲弶鍧婇崯绌傛鍏拷!k椤線绗ㄥЧ鏍埓閿燂拷5p椤炵穩閹鸿壈顫ㄩ崡顒冩緱|Q\\閻ュ濮圴閻︼拷.閿燂拷(Z韫囶喛顨夋い瀛樼毣h閽傚寒鍙曢敓锟�2椤庢换C閿燂拷)閸滆壈鐝滈惇缁㈠櫏&鏉岊噯鎷�*闁寸睜閿燂拷缁猴拷Uj濡插闈犻悰鍍欑憲鐤烽敓锟�:閸嬪骸娅桭闂侊拷閽勬稖婀掗埊娆愮箺{椤掓牞鏁撴锕傚灱S妫板儻閿燂拷8e椤栵拷.缁旀挸顑歗閿燂拷閸岃尪濮炵缓濠呮箑>                               <)閿熸枻鎷穥?闂冩帟鍎瘆閿燂拷$閿燂拷                                闁棜鍦洪敓锟�鐎涘啳鍤潂閻徏閸炲懘灏嘚閼哄獣閺嗙挙%缁斾粙鍓跺鏈�
閳躲倕鐣�5閿燂拷閾︻枮e\4妞村桓U鐟曪拷1璐稿畷姗�`閿燂拷"S'閿燂拷閽栨瓛QUD鐠ㄥ穲E鐟曪拷(閻欒偐锛濋崣鎾绘憙閸欙拷7濞堬拷x閸ゅ嫯寮粊宀冩窗閷呪晝鍎换锟�:閻瑧鏈濋悿鍐^妞存媭BR闁剧媭鎯欐锟�=瀹€鏉块兇_閿燂拷
_?闁惧掣妤ｃ劋婵€6^閹佃壈澹撻柗鍌�	k閻车椤栧⿵鎷�?g閻濓拷=椤愵剝鐟榌閸涙粏澧�(閹崇敿妤卞﹨顑�2>閿燂拷US9閿燂拷j闁诡噯鎷�閻愮珜婵犵ū閾擃搻U3婵繗妗2閼叉梹鐗揥閺佲偓椤х_Z 闁帮拷(閹硷拷%WR3閸栴溈閻燂拷閽勫函鎷�8閿燂拷)U閿燂拷S閾擄拷#,eWdY鐢絾褰块瀣摗fv閿燂拷<閿燂拷+婵亙鐭閿燂拷i妲庮偄顣烽悵鍡欐惍閿燂拷$I閽冨本娑祏濠曞浚妾稿▔鐧�4婵狅富妲�"濮楊亜鏆�瀹ヮ亷鎷�0閳煎唬&Ug椤ㄥ憡绻濈亸妤冪檨閸掓稒澹廼>閿燂拷+缁夐攱鍎ヽ缁茬闂婏拷閵囷拷閸ヨ精绶�+閻忕櫢鎷�#婵Κ椤犳ǹ绺块崵鈧�~椤犵珎"J閵婃柨銈锋Η顔剧槣闂€顐敳-閶湠娴狅箓鐭瀕妤ｅ牏顎為敓锟�4姒涚懓鐏╛閿燂拷  椤擄拷濞夋挴鍋�5 閿燂拷  