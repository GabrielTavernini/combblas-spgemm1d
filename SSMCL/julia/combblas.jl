module combblas
    using CxxWrap
    hostname = gethostname()
    if startswith(hostname, "login") || startswith(hostname,"nid") 
        @wrapmodule(() -> joinpath("/pscratch/sd/y/yuxihong/graphclustering/combblascorrect/SSMCL/","libjuliawrapper"))
    else
        @wrapmodule(() -> joinpath("/home/hongy0a/graphclustering/CombBLAS_YuxiPrivate/build/SSMCL/","libjuliawrapper"))
    end
    function __init__()
        @initcxx
    end
end