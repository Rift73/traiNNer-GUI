// Windows Script Host. Read the raw, adapter-wide WDDM dedicated memory counter.
var adapters = GetObject("winmgmts:root/cimv2").ExecQuery(
    "SELECT Name, DedicatedUsage FROM Win32_PerfRawData_GPUPerformanceCounters_GPUAdapterMemory"
);
for (var rows = new Enumerator(adapters); !rows.atEnd(); rows.moveNext()) {
    var adapter = rows.item();
    WScript.Echo(adapter.Name + " " + adapter.DedicatedUsage);
}
