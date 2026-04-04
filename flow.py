from cat import hook, CheshireCat, BillTheLizard


@hook(priority=0)
async def after_cheshire_cat_creation(cat: CheshireCat, lizard: BillTheLizard) -> None:
    this_plugin_id = lizard.mad_hatter.get_plugin().id
    await cat.plugin_manager.toggle_plugin(this_plugin_id)
