(function() {var implementors = {};
implementors["ansi_term"] = [{"text":"impl&lt;'a, S:&nbsp;'a + ToOwned + ?Sized&gt; Deref for <a class=\"struct\" href=\"ansi_term/struct.ANSIGenericString.html\" title=\"struct ansi_term::ANSIGenericString\">ANSIGenericString</a>&lt;'a, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;S as ToOwned&gt;::Owned: Debug,&nbsp;</span>","synthetic":false,"types":["ansi_term::display::ANSIGenericString"]}];
implementors["ascii"] = [{"text":"impl Deref for <a class=\"struct\" href=\"ascii/struct.AsciiString.html\" title=\"struct ascii::AsciiString\">AsciiString</a>","synthetic":false,"types":["ascii::ascii_string::AsciiString"]}];
implementors["bytes"] = [{"text":"impl Deref for <a class=\"struct\" href=\"bytes/struct.Bytes.html\" title=\"struct bytes::Bytes\">Bytes</a>","synthetic":false,"types":["bytes::bytes::Bytes"]},{"text":"impl Deref for <a class=\"struct\" href=\"bytes/struct.BytesMut.html\" title=\"struct bytes::BytesMut\">BytesMut</a>","synthetic":false,"types":["bytes::bytes_mut::BytesMut"]}];
implementors["cache_padded"] = [{"text":"impl&lt;T&gt; Deref for <a class=\"struct\" href=\"cache_padded/struct.CachePadded.html\" title=\"struct cache_padded::CachePadded\">CachePadded</a>&lt;T&gt;","synthetic":false,"types":["cache_padded::CachePadded"]}];
implementors["crossbeam_epoch"] = [{"text":"impl&lt;T:&nbsp;?Sized + <a class=\"trait\" href=\"crossbeam_epoch/trait.Pointable.html\" title=\"trait crossbeam_epoch::Pointable\">Pointable</a>&gt; Deref for <a class=\"struct\" href=\"crossbeam_epoch/struct.Owned.html\" title=\"struct crossbeam_epoch::Owned\">Owned</a>&lt;T&gt;","synthetic":false,"types":["crossbeam_epoch::atomic::Owned"]}];
implementors["crossbeam_utils"] = [{"text":"impl&lt;T&gt; Deref for <a class=\"struct\" href=\"crossbeam_utils/struct.CachePadded.html\" title=\"struct crossbeam_utils::CachePadded\">CachePadded</a>&lt;T&gt;","synthetic":false,"types":["crossbeam_utils::cache_padded::CachePadded"]},{"text":"impl&lt;T:&nbsp;?Sized&gt; Deref for <a class=\"struct\" href=\"crossbeam_utils/sync/struct.ShardedLockReadGuard.html\" title=\"struct crossbeam_utils::sync::ShardedLockReadGuard\">ShardedLockReadGuard</a>&lt;'_, T&gt;","synthetic":false,"types":["crossbeam_utils::sync::sharded_lock::ShardedLockReadGuard"]},{"text":"impl&lt;T:&nbsp;?Sized&gt; Deref for <a class=\"struct\" href=\"crossbeam_utils/sync/struct.ShardedLockWriteGuard.html\" title=\"struct crossbeam_utils::sync::ShardedLockWriteGuard\">ShardedLockWriteGuard</a>&lt;'_, T&gt;","synthetic":false,"types":["crossbeam_utils::sync::sharded_lock::ShardedLockWriteGuard"]}];
implementors["either"] = [{"text":"impl&lt;L, R&gt; Deref for <a class=\"enum\" href=\"either/enum.Either.html\" title=\"enum either::Either\">Either</a>&lt;L, R&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;L: Deref,<br>&nbsp;&nbsp;&nbsp;&nbsp;R: Deref&lt;Target = L::Target&gt;,&nbsp;</span>","synthetic":false,"types":["either::Either"]}];
implementors["managed"] = [{"text":"impl&lt;'a, T:&nbsp;'a + ?Sized&gt; Deref for <a class=\"enum\" href=\"managed/enum.Managed.html\" title=\"enum managed::Managed\">Managed</a>&lt;'a, T&gt;","synthetic":false,"types":["managed::object::Managed"]},{"text":"impl&lt;'a, T:&nbsp;'a&gt; Deref for <a class=\"enum\" href=\"managed/enum.ManagedSlice.html\" title=\"enum managed::ManagedSlice\">ManagedSlice</a>&lt;'a, T&gt;","synthetic":false,"types":["managed::slice::ManagedSlice"]}];
implementors["scopeguard"] = [{"text":"impl&lt;T, F, S&gt; Deref for <a class=\"struct\" href=\"scopeguard/struct.ScopeGuard.html\" title=\"struct scopeguard::ScopeGuard\">ScopeGuard</a>&lt;T, F, S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;F: FnOnce(T),<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"scopeguard/trait.Strategy.html\" title=\"trait scopeguard::Strategy\">Strategy</a>,&nbsp;</span>","synthetic":false,"types":["scopeguard::ScopeGuard"]}];
implementors["smoltcp"] = [{"text":"impl&lt;'a, T:&nbsp;Session&gt; Deref for <a class=\"struct\" href=\"smoltcp/socket/struct.SocketRef.html\" title=\"struct smoltcp::socket::SocketRef\">Ref</a>&lt;'a, T&gt;","synthetic":false,"types":["smoltcp::socket::ref_::Ref"]}];
implementors["tinyvec"] = [{"text":"impl&lt;A:&nbsp;<a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt; Deref for <a class=\"struct\" href=\"tinyvec/struct.ArrayVec.html\" title=\"struct tinyvec::ArrayVec\">ArrayVec</a>&lt;A&gt;","synthetic":false,"types":["tinyvec::arrayvec::ArrayVec"]},{"text":"impl&lt;'s, T&gt; Deref for <a class=\"struct\" href=\"tinyvec/struct.SliceVec.html\" title=\"struct tinyvec::SliceVec\">SliceVec</a>&lt;'s, T&gt;","synthetic":false,"types":["tinyvec::slicevec::SliceVec"]},{"text":"impl&lt;A:&nbsp;<a class=\"trait\" href=\"tinyvec/trait.Array.html\" title=\"trait tinyvec::Array\">Array</a>&gt; Deref for <a class=\"enum\" href=\"tinyvec/enum.TinyVec.html\" title=\"enum tinyvec::TinyVec\">TinyVec</a>&lt;A&gt;","synthetic":false,"types":["tinyvec::tinyvec::TinyVec"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()